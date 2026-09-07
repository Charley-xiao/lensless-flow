"""Cache a reproducible full-split RBC reconstruction comparison.

No target enters any reconstruction. Float32 predictions and per-image seeds
permit downstream metrics and visualizations without repeating inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from lensless_flow.data import HumanRBCHologramDataset
from lensless_flow.holography_offaxis import extract_offaxis_field, compensate_background
from lensless_flow.model_factory import build_flow_model, load_checkpoint_state_dict
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_latents(indices, shape, device, seed=20260903, stride=100003):
    """Use the same CUDA RNG seed as the existing batch-size-one evaluator."""
    return torch.cat([torch.randn((1, *shape), device=device,
                      generator=torch.Generator(device=device).manual_seed(seed + int(i) * stride))
                      for i in indices])


def offaxis_prediction(image):
    started = time.perf_counter()
    try:
        field = extract_offaxis_field(image, bandwidth=0.08, taper=0.015)
        result = compensate_background(field, border=16)
        prediction = result.pop('phase_png')
        details = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        details.update(status='reconstructed', bandwidth=0.08)
    except (ValueError, FloatingPointError) as exc:
        # Preserve full-set coverage, with explicit failure accounting. This
        # fixed zero-relative-phase fallback is not a successful reconstruction.
        prediction = np.full_like(image, 0.5, dtype=np.float32)
        details = dict(status='fallback_constant', error=str(exc), bandwidth=0.08)
    details['runtime_ms'] = 1000 * (time.perf_counter() - started)
    return prediction, details


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This full-model benchmark requires CUDA.')
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = HumanRBCHologramDataset(args.data_path, split='validation', downsample=1)
    n = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))
    if n < 1:
        raise ValueError('Need at least one pair.')
    specs = dict(item.split('=', 1) for item in args.model)
    if len(specs) != len(args.model) or any(not key.replace('_', '').isalnum() for key in specs):
        raise ValueError('Each model needs a unique alphanumeric/underscore name.')
    paths = [{'index': i, 'hologram': dataset.pairs[i][0].name, 'target': dataset.pairs[i][1].name}
             for i in range(n)]
    manifest = dict(dataset_path=args.data_path, split='Validation', dataset_total=len(dataset), samples=n,
        selection='all sorted paired filenames' if n == len(dataset) else 'smoke test: first N pairs',
        models={name: {'path': path, 'sha256': sha256(path)} for name, path in specs.items()},
        input_only_physics={'bandwidth': 0.08, 'taper': 0.015, 'border': 16,
                            'failure_policy': 'explicitly counted constant-0.5 fallback'},
        seed=args.seed, seed_stride=100003, solver='heun', steps=args.steps,
        precision='float32 with TF32 enabled, no AMP', batch_size=args.batch_size,
        image_shape=[1, 256, 256], prediction_storage='float32 unclamped; metrics clamp to [0,1]',
        gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__), paths=paths)
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise ValueError('Output manifest differs. Choose a new output directory for changed settings.')
    else:
        write_json(manifest_path, manifest)
    progress_path = out/'reconstruction_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'next_index': 0, 'seconds': 0.0}
    start_index = progress['next_index']
    if start_index == n:
        print('Reconstruction cache already complete.', flush=True)
        return
    predictions = {}
    for name in [*specs, 'offaxis_input_only']:
        path = out/(name + '.npy')
        if start_index and not path.exists():
            raise ValueError(f'Missing cache for resumed method: {name}')
        predictions[name] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                                      dtype=np.float32, shape=(n, 1, 256, 256))
    models = {}
    for name, path in specs.items():
        state = torch.load(path, map_location='cpu', weights_only=True)
        cfg = state['cfg']
        if cfg['train']['mode'] != 'vanilla' or cfg['cfm'].get('source', {}).get('mode', 'gaussian') != 'gaussian':
            raise ValueError('This protocol is for vanilla Gaussian-source flow checkpoints.')
        if float(cfg['cfm'].get('source', {}).get('sigma0', 1)) != 1:
            raise ValueError('Expected Gaussian source sigma0=1.')
        if any(cfg['data'].get(k, False) for k in ['phase_invert','hologram_invert','flip_ud','flip_lr']):
            raise ValueError('This comparison expects the audited non-inverted, non-flipped dataset.')
        model = build_flow_model(cfg, 1, (256, 256), device, checkpoint_state=state)
        load_checkpoint_state_dict(model, state)
        models[name] = model.eval()
        print(f'Loaded {name}: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
        del state

    def sample(model, y, initial):
        return sample_with_physics_guidance(model, y, H=None, steps=args.steps,
            dc_step=0, dc_steps=0, init_noise_std=1, clamp_x=False,
            disable_physics=True, pred_type='vanilla', solver='heun', initial_state=initial)

    # Warm up without targets and verify that injecting the fixed latent agrees
    # with the original stochastic sampler for one seed and one image.
    warm_y = to_nchw(dataset[0][0]).to(device)
    first_model = next(iter(models.values()))
    torch.manual_seed(args.seed)
    legacy = sample_with_physics_guidance(first_model, warm_y, H=None, steps=args.steps,
        dc_step=0, dc_steps=0, init_noise_std=1, clamp_x=False,
        disable_physics=True, pred_type='vanilla', solver='heun')
    injected = sample(first_model, warm_y, image_latents([0], (1,256,256), device, args.seed))
    difference = float((legacy - injected).abs().max())
    if difference > 1e-6:
        raise RuntimeError(f'Explicit latent differs from original sampler: {difference}')
    write_json(out/'sampler_equivalence.json', {'index': 0, 'max_abs_difference': difference,
                                               'steps': args.steps, 'seed': args.seed})
    del legacy, injected, warm_y
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.physics_workers) as pool, (out/'reconstruction_batches.jsonl').open('a') as log:
        for offset in range(start_index, n, args.batch_size):
            end = min(offset + args.batch_size, n)
            indices = list(range(offset, end))
            # Only the hologram is passed to either learned or physical inference.
            holograms = torch.stack([dataset[i][0] for i in indices]).permute(0,3,1,2).contiguous()
            futures = [pool.submit(offaxis_prediction, image[0].numpy()) for image in holograms]
            y = holograms.to(device)
            initial = image_latents(indices, (1,256,256), device, args.seed)
            durations, clipping = {}, {}
            for name, model in models.items():
                torch.cuda.synchronize()
                before = time.perf_counter()
                pred = sample(model, y, initial)
                torch.cuda.synchronize()
                durations[name] = 1000 * (time.perf_counter() - before) / len(indices)
                if not torch.isfinite(pred).all():
                    raise RuntimeError(f'Nonfinite prediction: {name}, batch {offset}')
                clipping[name] = ((pred < 0) | (pred > 1)).float().mean((1,2,3)).cpu().tolist()
                predictions[name][offset:end] = pred.cpu().numpy()
            physics = [future.result() for future in futures]
            predictions['offaxis_input_only'][offset:end] = np.stack([item[0][None] for item in physics])
            for array in predictions.values():
                array.flush()
            log.write(json.dumps({'start': offset, 'end': end, 'flow_ms_per_image': durations,
                'clipped_fraction': clipping, 'physics': [item[1] for item in physics]}) + '\n')
            log.flush()
            elapsed = progress['seconds'] + time.perf_counter() - started
            write_json(progress_path, {'next_index': end, 'total': n, 'seconds': elapsed})
            print(json.dumps({'stage': 'reconstruction', 'done': end, 'total': n,
                              'seconds': round(elapsed,1), 'flow_ms_per_image': durations}), flush=True)
    print('Full reconstruction cache complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model', action='append', required=True, help='NAME=/absolute/checkpoint.pt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--physics_workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20260903)
    parser.add_argument('--max_samples', type=int, default=-1)
    main(parser.parse_args())
