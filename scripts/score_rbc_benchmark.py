"""Score every cached RBC reconstruction with identical full-split metrics."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

from lensless_flow.data import HumanRBCHologramDataset
from lensless_flow.metrics import ssim_map_torch, region_image_metrics
from lensless_flow.rbc_regions import rbc_region_mask
from scripts.benchmark_rbc_reconstruction import write_json


def pixel_metrics(pred, target, mask):
    """Per-image metrics; input images are already clamped, with range one."""
    squared = (pred - target).square()
    mse = squared.mean((1,2,3))
    circular = torch.atan2(torch.sin(2 * torch.pi * (pred-target)), torch.cos(2 * torch.pi * (pred-target)))

    def from_mse(value):
        return torch.where(value <= 1e-12, 99.0, -10 * torch.log10(value.clamp_min(1e-12)))

    crop_p, crop_t = pred[...,16:-16,16:-16], target[...,16:-16,16:-16]
    result = dict(mse=mse, rmse=mse.sqrt(), mae=(pred-target).abs().mean((1,2,3)),
        psnr=from_mse(mse), ssim=ssim_map_torch(pred,target).mean((1,2,3)),
        circular_rmse_rad=circular.square().mean((1,2,3)).sqrt(),
        interior_psnr=from_mse((crop_p-crop_t).square().mean((1,2,3))),
        interior_ssim=ssim_map_torch(crop_p,crop_t).mean((1,2,3)))
    result.update(region_image_metrics(pred, target, mask))
    result['rbc_rmse'] = result['rbc_mse'].sqrt()
    result['background_rmse'] = result['background_mse'].sqrt()
    background_count = (1-mask).sum((1,2,3))
    background_circular = (circular.square() * (1-mask)).sum((1,2,3)) / background_count.clamp_min(1)
    result['background_circular_rmse_rad'] = torch.where(background_count > 0, background_circular.sqrt(), torch.nan)
    return result


def aggregate(rows):
    result = {}
    excluded = {'index', 'hologram', 'target', 'method', 'physics_status'}
    for key in rows[0]:
        if key in excluded:
            continue
        values = np.array([float(row[key]) for row in rows if row[key] not in (None, '')], dtype=float)
        values = values[np.isfinite(values)]
        result[key] = dict(n=int(len(values)), mean=float(values.mean()) if len(values) else None,
            std=float(values.std(ddof=1)) if len(values)>1 else None,
            median=float(np.median(values)) if len(values) else None,
            p05=float(np.quantile(values,.05)) if len(values) else None,
            p95=float(np.quantile(values,.95)) if len(values) else None)
    return result


@torch.inference_mode()
def main(args):
    import lpips
    from torchmetrics.image.fid import FrechetInceptionDistance

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(args.cache_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    n = manifest['samples']
    progress = json.loads((root/'reconstruction_progress.json').read_text())
    if progress['next_index'] != n:
        raise ValueError('Reconstruction cache must cover every selected image before scoring.')
    dataset = HumanRBCHologramDataset(manifest['dataset_path'], split='validation', downsample=1)
    if len(dataset) != manifest['dataset_total'] or any(dataset.pairs[i][0].name != item['hologram']
                                                       or dataset.pairs[i][1].name != item['target']
                                                       for i,item in enumerate(manifest['paths'])):
        raise ValueError('Dataset pairing/order changed since inference.')
    methods = [*manifest['models'], 'offaxis_input_only', 'measurement', 'constant_0_5']
    out = root/'metrics'
    out.mkdir(exist_ok=True)
    timings = {name: np.full(n, np.nan) for name in methods}
    physics_status = ['unknown'] * n
    for line in (root/'reconstruction_batches.jsonl').read_text().splitlines():
        record = json.loads(line)
        begin, end = record['start'], record['end']
        for name, runtime in record['flow_ms_per_image'].items():
            timings[name][begin:end] = runtime
        timings['offaxis_input_only'][begin:end] = [item['runtime_ms'] for item in record['physics']]
        physics_status[begin:end] = [item['status'] for item in record['physics']]

    perceptual = lpips.LPIPS(net='alex', version='0.1').to(device).eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device).eval()
    if fid.real_features_sum.dtype != torch.float64:
        raise RuntimeError('Expected float64 FID statistical accumulators.')
    notes = dict(samples=n, methods=methods, metric_device=str(device), metric_batch_size=args.batch_size,
        lpips='AlexNet v0.1; grayscale repeated to RGB, normalized to [-1,1]',
        fid='Inception 2048; grayscale repeated to RGB, [0,1] floats converted to uint8 by TorchMetrics; float64 moments',
        pixel_metrics='Clamp every method to [0,1], fixed range 1; arithmetic means of per-image scores',
        ssim='Repository 11x11 Gaussian sigma 1.5, zero padding; regional SSIM selects map centers',
        interior='Remove 16 pixels on every side, then compute PSNR and SSIM on the cropped images',
        circular='2*pi radians per normalized image interval; no target-assisted gauge/sign/tilt fitting',
        regions='Frozen target pseudo-masks: circular window15, threshold .08, closing5, padding48, component80, coverage .01-.85',
        missing_regions='NaN excluded only from that regional score; valid counts reported',
        runtime='Flow: synchronized batch inference wall-time divided by actual batch size, warmup excluded; physics: individual CPU call wall-time with concurrent workers. Excludes metrics and data loading.',
        dc_rmse=None, dc_rmse_reason='No calibrated common forward operator maps the normalized phase PNG alone to recorded intensity (unknown reference, amplitude and export scale).',
        physics_fallbacks=physics_status.count('fallback_constant'),
        versions={key: importlib.metadata.version(key) for key in ['torch','torchmetrics','torch-fidelity','lpips','scikit-image','scipy','numpy']})
    write_json(out/'metric_protocol.json', notes)
    mask_path = root/'target_regions.npy'
    masks = np.lib.format.open_memmap(mask_path, mode='w+', dtype=np.uint8, shape=(n,1,256,256))
    # Compute the target distribution once. reset_real_features=False retains
    # it across methods, and each fake distribution has the same N images.
    for begin in range(0,n,args.batch_size):
        end = min(begin+args.batch_size,n)
        target = torch.stack([dataset[i][1] for i in range(begin,end)]).permute(0,3,1,2).contiguous()
        masks[begin:end] = rbc_region_mask(target).numpy().astype(np.uint8)
        fid.update(target.to(device).repeat(1,3,1,1), real=True)
        if begin % (args.batch_size*32)==0 or end==n:
            print(json.dumps({'stage':'target_statistics','done':end,'total':n}),flush=True)
    masks.flush()
    summary = {}
    for method in methods:
        summary_path = out/(method+'_summary.json')
        if summary_path.exists():
            summary[method] = json.loads(summary_path.read_text())
            continue
        predictions = None if method in ['measurement','constant_0_5'] else np.load(root/(method+'.npy'), mmap_mode='r')
        rows = []
        fid.reset()
        started = time.perf_counter()
        with (out/(method+'_per_image.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = None
            for begin in range(0,n,args.batch_size):
                end = min(begin+args.batch_size,n)
                pairs = [dataset[i] for i in range(begin,end)]
                target = torch.stack([pair[1] for pair in pairs]).permute(0,3,1,2).contiguous().to(device)
                if method == 'measurement':
                    raw = torch.stack([pair[0] for pair in pairs]).permute(0,3,1,2).contiguous().to(device)
                elif method == 'constant_0_5':
                    raw = torch.full_like(target, 0.5)
                else:
                    raw = torch.from_numpy(np.array(predictions[begin:end])).to(device)
                if not torch.isfinite(raw).all():
                    raise ValueError(f'Nonfinite cached prediction: {method}, {begin}')
                clipped_fraction = ((raw<0)|(raw>1)).float().mean((1,2,3))
                pred = raw.clamp(0,1)
                target = target.clamp(0,1)
                mask = torch.from_numpy(np.array(masks[begin:end],dtype=np.float32)).to(device)
                values = pixel_metrics(pred,target,mask)
                values['clipped_fraction'] = clipped_fraction
                values['lpips'] = perceptual(pred.repeat(1,3,1,1)*2-1, target.repeat(1,3,1,1)*2-1).flatten()
                fid.update(pred.repeat(1,3,1,1),real=False)
                cpu = {key:value.cpu().tolist() for key,value in values.items()}
                for j,index in enumerate(range(begin,end)):
                    row = {**manifest['paths'][index], 'method':method,
                        **{key:val[j] if np.isfinite(val[j]) else None for key,val in cpu.items()},
                        'runtime_ms':float(timings[method][index]) if np.isfinite(timings[method][index]) else None,
                        'physics_status':physics_status[index] if method=='offaxis_input_only' else ''}
                    if writer is None:
                        writer = csv.DictWriter(handle,fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows.append(row)
                handle.flush()
                if begin % (args.batch_size*16)==0 or end==n:
                    print(json.dumps({'stage':'metrics','method':method,'done':end,'total':n,
                                      'seconds':round(time.perf_counter()-started,1)}),flush=True)
        counts = (int(fid.real_features_num_samples),int(fid.fake_features_num_samples))
        if counts != (n,n):
            raise RuntimeError(f'Incomplete FID sample counts: {counts}')
        fid_value = float(fid.compute())
        result = dict(method=method,samples=len(rows),fid=fid_value,fid_real_samples=counts[0],
                      fid_fake_samples=counts[1],metrics=aggregate(rows))
        write_json(summary_path,result)
        summary[method]=result
        write_json(out/'summary.json',summary)
        print(json.dumps({'method_complete':method,'samples':len(rows),'fid':fid_value,
            'psnr':result['metrics']['psnr']['mean'],'ssim':result['metrics']['ssim']['mean']}),flush=True)
    write_json(out/'summary.json',summary)
    flat = [{'method':name,'samples':item['samples'],'fid':item['fid'],
              **{key:stat['mean'] for key,stat in item['metrics'].items()}} for name,item in summary.items()]
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write_json(out/'complete.json',{'samples_per_method':n,'methods':methods,'status':'complete'})
    print('All metrics complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir',required=True)
    parser.add_argument('--batch_size',type=int,default=32)
    main(parser.parse_args())
