"""Add a direct U-Net to an audited full-validation cache and time all methods.

Existing flow/physics predictions and scores are reused only after matching
dataset order, model hashes, metric code and package versions. No labels enter
reconstruction. Separate matched timing excludes I/O and quality metrics.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from lensless_flow.data import HumanRBCHologramDataset
from lensless_flow.model_factory import build_baseline_unet, build_flow_model, load_checkpoint_state_dict
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.sampler import sample_with_physics_guidance
from scripts.benchmark_rbc_reconstruction import image_latents, offaxis_prediction, sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def function_ast(path, name):
    node = next(node for node in ast.parse(Path(path).read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    return ast.dump(node, include_attributes=False)


def direct(model, y):
    return model(torch.zeros_like(y), y, None)


def flow(model, y, indices, manifest):
    initial = image_latents(indices, (1, 256, 256), y.device, manifest['seed'], manifest['seed_stride'])
    return sample_with_physics_guidance(model, y, H=None, steps=manifest['steps'],
        dc_step=0, dc_steps=0, init_noise_std=1, clamp_x=False, disable_physics=True,
        pred_type='vanilla', solver=manifest['solver'], initial_state=initial)


def load_direct(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state.get('prediction_type') != 'direct_image' or state.get('use_time_conditioning') is not False
            or state.get('training_loss') != 'mse'):
        raise ValueError('Expected the deterministic image-MSE U-Net checkpoint.')
    model = build_baseline_unet(state['cfg'], 1, device, checkpoint_state=state)
    load_checkpoint_state_dict(model, state)
    return model, state


def verify_reference(reference, dataset):
    reference = Path(reference)
    manifest = read_json(reference/'manifest.json')
    complete = read_json(reference/'metrics/complete.json')
    protocol = read_json(reference/'metrics/metric_protocol.json')
    if (manifest['samples'] != len(dataset) or manifest['dataset_total'] != len(dataset)
            or complete['status'] != 'complete' or complete['samples_per_method'] != len(dataset)):
        raise ValueError('Reference must be a completed full validation benchmark.')
    paths = [{'index': i, 'hologram': pair[0].name, 'target': pair[1].name}
             for i, pair in enumerate(dataset.pairs)]
    if paths != manifest['paths']:
        raise ValueError('Pairing/order differs from the historical benchmark.')
    source_root = reference.parents[1]
    for name in ['lensless_flow/metrics.py', 'lensless_flow/holography_offaxis.py',
                 'lensless_flow/sampler.py', 'scripts/score_rbc_benchmark.py',
                 'scripts/benchmark_rbc_reconstruction.py', 'lensless_flow/data.py']:
        if sha256(name) != sha256(source_root/name):
            raise ValueError(f'Reference implementation changed: {name}')
    if function_ast('lensless_flow/rbc_regions.py', 'rbc_region_mask') != function_ast(
            source_root/'lensless_flow/rbc_regions.py', 'rbc_region_mask'):
        raise ValueError('RBC mask computation changed.')
    for package, expected in protocol['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Metric dependency changed: {package}')
    if manifest['gpu'] != torch.cuda.get_device_name():
        raise ValueError('GPU differs from reference.')
    return manifest


@torch.inference_mode()
def cache(args):
    out, reference = Path(args.out_dir), Path(args.reference_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Use a fresh output directory; do not overwrite an evaluation.')
    dataset = HumanRBCHologramDataset(args.data_path, split='validation', downsample=1)
    manifest = verify_reference(reference, dataset)
    model, state = load_direct(args.checkpoint, 'cuda')
    records = [json.loads(line) for line in Path(args.training_metrics).read_text().splitlines()]
    best = max(records[1:], key=lambda row: row.get('eval/rbc_ssim') or -1)
    if state['epoch'] != best['epoch'] or state['eval_metrics']['eval/rbc_ssim'] != best['eval/rbc_ssim']:
        raise ValueError('Saved checkpoint does not match the selected best epoch.')
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), sha256=sha256(args.checkpoint),
        best_epoch=state['epoch'], completed_epoch=records[-1]['epoch'], interrupted_epoch=records[-1]['epoch']+1,
        selection_metric='eval/rbc_ssim', selection_samples=128, selection_score=best['eval/rbc_ssim'],
        global_ssim_best_epoch=max(records[1:],key=lambda row:row['eval/ssim'])['epoch'],
        checkpoint_metrics=state['eval_metrics'], stopped_by_user=True)
    write_json(out/'checkpoint_selection.json', metadata)
    manifest['models'] = {'direct_unet_best': {'path':metadata['checkpoint'], 'sha256':metadata['sha256']},
                          **manifest['models']}
    manifest['reference_cache'] = str(reference.resolve())
    manifest['reused_methods'] = ['scratch_best', 'offaxis_input_only', 'measurement', 'constant_0_5']
    manifest['model_types'] = {'direct_unet_best':'deterministic supervised image regression', 'scratch_best':'flow'}
    manifest['evaluated_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(out/'manifest.json', manifest)
    metrics_dir = out/'metrics'
    metrics_dir.mkdir()
    for method in manifest['reused_methods']:
        for suffix in ['_summary.json','_per_image.csv']:
            shutil.copy2(reference/'metrics'/(method+suffix), metrics_dir/(method+suffix))
        if method not in ['measurement','constant_0_5']:
            # Read-only reference link; this script never writes reused predictions.
            (out/(method+'.npy')).symlink_to(reference/(method+'.npy'))
    write_json(out/'reuse_provenance.json', {
        'reference':str(reference), 'verified':'full coverage, filenames, inference/metric source, mask AST, metric versions, GPU',
        'reused_file_sha256':{str(path.relative_to(reference)):sha256(path)
            for path in [reference/'manifest.json', reference/'metrics/summary.json',
                         reference/'scratch_best.npy', reference/'offaxis_input_only.npy']}})
    n, batch = len(dataset), manifest['batch_size']
    prediction = np.lib.format.open_memmap(out/'direct_unet_best.npy', mode='w+', dtype=np.float32,
                                          shape=(n,1,256,256))
    warm = torch.stack([dataset[i][0] for i in range(batch)]).permute(0,3,1,2).contiguous().cuda()
    for _ in range(3):
        direct(model, warm)
    torch.cuda.synchronize()
    del warm
    old_records = [json.loads(line) for line in (reference/'reconstruction_batches.jsonl').read_text().splitlines()]
    started = time.perf_counter()
    with (out/'reconstruction_batches.jsonl').open('w') as handle:
        for offset in range(0, n, batch):
            end = min(offset+batch,n)
            y = torch.stack([dataset[i][0] for i in range(offset,end)]).permute(0,3,1,2).contiguous().cuda()
            torch.cuda.synchronize()
            before = time.perf_counter()
            pred = direct(model,y)
            torch.cuda.synchronize()
            elapsed = 1000*(time.perf_counter()-before)/(end-offset)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f'Nonfinite U-Net prediction at {offset}.')
            prediction[offset:end] = pred.cpu().numpy()
            old = old_records[offset//batch]
            if (old['start'],old['end']) != (offset,end):
                raise ValueError('Reference batch boundaries differ.')
            old['flow_ms_per_image']['direct_unet_best'] = elapsed  # Legacy timing-field name.
            old['clipped_fraction']['direct_unet_best'] = ((pred<0)|(pred>1)).float().mean((1,2,3)).cpu().tolist()
            handle.write(json.dumps(old)+'\n')
            if offset % (batch*32)==0 or end==n:
                print(json.dumps({'stage':'direct_inference','done':end,'total':n,'ms_per_image':elapsed}),flush=True)
        prediction.flush()
    seconds = time.perf_counter()-started
    write_json(out/'reconstruction_progress.json', {'next_index':n,'total':n,'seconds':seconds})
    write_json(out/'direct_evaluation_wall_time.json', {
        'inference_cache_wall_seconds':seconds,'inference_cache_wall_ms_per_image':1000*seconds/n,
        'note':'Includes paired PNG loading, transfers, reconstruction and float32 cache writes; excludes warmup, model loading and quality metrics.'})
    print('Direct U-Net full-validation cache complete.',flush=True)


@torch.inference_mode()
def timing(args):
    out = Path(args.out_dir)
    manifest = read_json(out/'manifest.json')
    dataset = HumanRBCHologramDataset(manifest['dataset_path'], split='validation', downsample=1)
    device = torch.device('cuda')
    unet, _ = load_direct(manifest['models']['direct_unet_best']['path'], device)
    flow_path = manifest['models']['scratch_best']['path']
    if sha256(flow_path) != manifest['models']['scratch_best']['sha256']:
        raise ValueError('Flow checkpoint changed since the reference evaluation.')
    state = torch.load(flow_path,map_location='cpu',weights_only=True)
    flow_model = build_flow_model(state['cfg'],1,(256,256),device,checkpoint_state=state)
    load_checkpoint_state_dict(flow_model,state)
    models = {'direct_unet_best':unet,'scratch_best':flow_model}
    def predict(name,y,indices):
        return direct(models[name],y) if name=='direct_unet_best' else flow(models[name],y,indices,manifest)
    # Confirm that the reused physics and flow cache still represent the executed methods.
    indices = list(range(manifest['batch_size']))
    y = torch.stack([dataset[i][0] for i in indices]).permute(0,3,1,2).contiguous().to(device)
    reused = np.load(out/'scratch_best.npy',mmap_mode='r')
    discrepancy = float((predict('scratch_best',y,indices).cpu()-torch.from_numpy(np.array(reused[:len(indices)]))).abs().max())
    if discrepancy > 1e-4:
        raise ValueError(f'Reused flow predictions disagree with matching fresh inference: {discrepancy}')
    physical = np.load(out/'offaxis_input_only.npy',mmap_mode='r')
    physics_differences=[]
    for i in np.linspace(0,len(dataset)-1,8,dtype=int):
        pred,_ = offaxis_prediction(dataset[int(i)][0].numpy()[...,0])
        physics_differences.append(float(np.max(np.abs(pred-physical[i,0]))))
    if max(physics_differences)>1e-5:
        raise ValueError('Reused physics predictions disagree with fresh inference.')
    write_json(out/'cache_reproduction_check.json', {'flow_first_batch_max_abs_difference':discrepancy,
        'physics_eight_evenly_spaced_max_abs_differences':physics_differences})
    selected = np.linspace(0,len(dataset)-1,args.timing_samples,dtype=int).tolist()
    inputs = torch.stack([dataset[i][0] for i in selected]).permute(0,3,1,2).contiguous()
    timings = {name:[] for name in [*models,'offaxis_input_only']}
    for name in models:
        warm=inputs[:1].to(device)
        for _ in range(3): predict(name,warm,selected[:1])
    for _ in range(3): offaxis_prediction(inputs[0,0].numpy())
    torch.cuda.synchronize()
    for j,index in enumerate(selected):
        y=inputs[j:j+1].to(device)
        names=list(timings)
        names=names[j%3:]+names[:j%3]  # Rotate order; no simultaneous method execution.
        for name in names:
            torch.cuda.synchronize()
            before=time.perf_counter()
            if name=='offaxis_input_only': offaxis_prediction(inputs[j,0].numpy())
            else: predict(name,y,[index])
            torch.cuda.synchronize()
            timings[name].append({'index':index,'ms':1000*(time.perf_counter()-before)})
        if (j+1)%16==0: print(json.dumps({'stage':'single_image_timing','done':j+1,'total':len(selected)}),flush=True)
    batched={name:[] for name in models}
    for name in models:
        predict(name,inputs[:16].to(device),selected[:16])
    torch.cuda.synchronize()
    for begin in range(0,len(selected),16):
        end=min(begin+16,len(selected))
        y=inputs[begin:end].to(device)
        for name in models:
            torch.cuda.synchronize()
            before=time.perf_counter()
            predict(name,y,selected[begin:end])
            torch.cuda.synchronize()
            ms=1000*(time.perf_counter()-before)/(end-begin)
            batched[name].extend([ms]*(end-begin))
        print(json.dumps({'stage':'batch16_timing','done':end,'total':len(selected)}),flush=True)
    def stats(values):
        return {'n':len(values),'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),
                'p05_ms':float(np.quantile(values,.05)),'p95_ms':float(np.quantile(values,.95))}
    result={'measured_utc':datetime.now(timezone.utc).isoformat(),'indices':selected,
            'selection':'128 evenly spaced sorted validation indices, independent of quality',
            'hardware':{'gpu':torch.cuda.get_device_name(),'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].split(':')[-1].strip(),
                        'torch_cpu_threads':torch.get_num_threads()},
            'precision':manifest['precision'],
            'protocol':'Three single-image warmups; synchronized serial inference with rotating method order. Inputs already in memory/on device. Excludes PNG I/O, host-device transfer, quality metrics and warmup. Flow includes image-specific latent generation and Heun40. CPU physics uses one call at a time; neural models use the same GPU.',
            'single_image':{name:stats([v['ms'] for v in rows]) for name,rows in timings.items()},
            'batch16_amortized':{name:stats(values) for name,values in batched.items()},'single_image_raw':timings}
    write_json(out/'matched_timing.json',result)
    print(json.dumps({'timing_complete':result['single_image']}),flush=True)


def export(args):
    out=Path(args.out_dir)
    manifest=read_json(out/'manifest.json')
    dataset=HumanRBCHologramDataset(manifest['dataset_path'],split='validation',downsample=1)
    indices=np.linspace(0,len(dataset)-1,8,dtype=int).tolist()
    methods=['offaxis_input_only','direct_unet_best','scratch_best']
    predictions={name:np.load(out/(name+'.npy'),mmap_mode='r') for name in methods}
    samples=np.stack([np.stack([dataset[i][0].numpy()[...,0],dataset[i][1].numpy()[...,0]]+
                              [predictions[name][i,0] for name in methods]) for i in indices])
    np.savez_compressed(out/'paper_samples.npz',images=samples,indices=np.array(indices))
    write_json(out/'paper_samples.json',{'indices':indices,'paths':[manifest['paths'][i] for i in indices],
        'columns':['Input hologram','Ground truth','Physics-informed','Direct U-Net','Flow'],
        'selection':'Eight evenly spaced sorted validation indices, fixed independently of reconstruction quality',
        'display_range':[0,1],'source_array_dtype':'float32','spatial_shape':[256,256]})
    print('Exported eight paired samples for the paper figure.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['cache','timing','export'])
    parser.add_argument('--out_dir',required=True)
    parser.add_argument('--reference_dir')
    parser.add_argument('--checkpoint')
    parser.add_argument('--training_metrics')
    parser.add_argument('--data_path',default='/home/qiwen/data/rbc_holograms_osf')
    parser.add_argument('--timing_samples',type=int,default=128)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    {'cache':cache,'timing':timing,'export':export}[args.stage](args)
