"""Generate full-split tables and fixed-scale visual comparisons from caches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lensless_flow.data import HumanRBCHologramDataset
from scripts.benchmark_rbc_reconstruction import write_json


LABELS = {'original_e105':'Original flow, epoch 105', 'original_e95':'Original flow, epoch 95',
          'region_final':'Modified-loss flow, final', 'offaxis_input_only':'Off-axis physics',
          'measurement':'Raw hologram', 'constant_0_5':'Constant 0.5'}


def number(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root=Path(args.cache_dir)
    manifest=json.loads((root/'manifest.json').read_text())
    completed=json.loads((root/'metrics/complete.json').read_text())
    summary=json.loads((root/'metrics/summary.json').read_text())
    protocol=json.loads((root/'metrics/metric_protocol.json').read_text())
    n=manifest['samples']
    methods=completed['methods']
    if completed['samples_per_method'] != n or any(summary[name]['samples'] != n for name in methods):
        raise ValueError('Report requires the same complete image set for every method.')
    report=root/'report'
    report.mkdir(exist_ok=True)
    dataset=HumanRBCHologramDataset(manifest['dataset_path'],split='validation',downsample=1)
    predictions={name:np.load(root/(name+'.npy'),mmap_mode='r') for name in methods if name not in ['measurement','constant_0_5']}
    selected=np.linspace(0,n-1,min(8,n),dtype=int).tolist()
    write_json(report/'visual_indices.json',{'selection':'evenly spaced sorted indices, independent of performance','indices':selected})
    image_links=[]
    visual_methods=[name for name in ['original_e105','original_e95','region_final','offaxis_input_only'] if name in methods]
    for page,begin in enumerate(range(0,len(selected),4),start=1):
        indices=selected[begin:begin+4]
        fig,axes=plt.subplots(len(indices),len(visual_methods)+2,figsize=(18,3*len(indices)),squeeze=False,layout='constrained')
        for row,index in enumerate(indices):
            y,x=dataset[index]
            images=[y.numpy()[...,0],x.numpy()[...,0]]+[predictions[name][index,0] for name in visual_methods]
            titles=['Hologram','Phase target']+[LABELS[name] for name in visual_methods]
            for col,(image,title) in enumerate(zip(images,titles)):
                ax=axes[row,col]
                ax.imshow(image,cmap='gray',vmin=0,vmax=1)
                if row==0: ax.set_title(title,fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
                if col==0: ax.set_ylabel(f'Validation index {index}')
        name=f'comparison_{page}.png'
        fig.savefig(report/name,dpi=140);plt.close(fig)
        image_links.append(f'![Fixed-scale comparison page {page}]({name})')

    panels=[('psnr','Global PSNR (dB)',False),('ssim','Global SSIM',False),
            ('rbc_psnr','RBC pseudo-region PSNR (dB)',False),('rbc_ssim','RBC pseudo-region SSIM',False),
            ('lpips','LPIPS (lower is better)',False),('fid','FID (lower is better)',True)]
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    short=['Flow e105','Flow e95','Region loss','Physics','Hologram','Flat 0.5']
    for ax,(key,title,dataset_metric) in zip(axes.flat,panels):
        values=[summary[name]['fid'] if dataset_metric else summary[name]['metrics'][key]['mean'] for name in methods]
        ax.bar(np.arange(len(methods)),values,color=['#5a7fa5','#7b97b5','#bd643e','#378673','#aaa7a3','#c8c5c1'])
        ax.set_xticks(np.arange(len(methods)),short,rotation=35,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(report/'metric_comparison.png',dpi=150);plt.close(fig)

    all_rows={}
    for name in methods:
        with (root/'metrics'/(name+'_per_image.csv')).open(encoding='utf-8') as handle:
            all_rows[name]=list(csv.DictReader(handle))
        if [int(row['index']) for row in all_rows[name]] != list(range(n)):
            raise ValueError(f'Incomplete or misordered per-image CSV: {name}')
    paired=[]
    for base in ['original_e105','original_e95','offaxis_input_only']:
        if base not in methods or 'region_final' not in methods: continue
        for key,higher in [('psnr',True),('ssim',True),('rbc_psnr',True),('rbc_ssim',True),('lpips',False),('rbc_circular_rmse_rad',False)]:
            differences=[]
            for candidate,reference in zip(all_rows['region_final'],all_rows[base]):
                if candidate[key] and reference[key]:
                    differences.append(float(candidate[key])-float(reference[key]))
            values=np.asarray(differences)
            wins=values>0 if higher else values<0
            paired.append(dict(candidate='region_final',reference=base,metric=key,n=len(values),
                mean_difference=float(values.mean()) if len(values) else None,
                median_difference=float(np.median(values)) if len(values) else None,
                win_fraction=float(wins.mean()) if len(values) else None))
    write_json(report/'paired_differences.json',paired)
    with (report/'paired_differences.csv').open('w',newline='',encoding='utf-8') as handle:
        if paired:
            writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)

    global_rows=[];region_rows=[];runtime_rows=[]
    for name in methods:
        item=summary[name]; m=item['metrics']
        global_rows.append([LABELS[name],n,*[number(m[key]['mean']) for key in ['psnr','ssim','mae','rmse','lpips']],number(item['fid'],3)])
        region_rows.append([LABELS[name],m['rbc_psnr']['n'],*[number(m[key]['mean']) for key in ['rbc_psnr','rbc_ssim','rbc_circular_rmse_rad','background_psnr','background_ssim']]])
        runtime_rows.append([LABELS[name],*[number(m[key]['mean']) for key in ['interior_psnr','interior_ssim','circular_rmse_rad','runtime_ms','clipped_fraction']]])
    best_global=max(methods,key=lambda name:summary[name]['metrics']['ssim']['mean'])
    best_rbc=max(methods,key=lambda name:summary[name]['metrics']['rbc_ssim']['mean'])
    observations=(f"The highest mean global SSIM is achieved by **{LABELS[best_global]}** "
                  f"({number(summary[best_global]['metrics']['ssim']['mean'])}); "
                  f"the highest mean RBC pseudo-region SSIM is achieved by **{LABELS[best_rbc]}** "
                  f"({number(summary[best_rbc]['metrics']['rbc_ssim']['mean'])}).")
    if 'region_final' in summary and 'original_e105' in summary:
        delta_psnr=summary['region_final']['metrics']['rbc_psnr']['mean']-summary['original_e105']['metrics']['rbc_psnr']['mean']
        delta_ssim=summary['region_final']['metrics']['rbc_ssim']['mean']-summary['original_e105']['metrics']['rbc_ssim']['mean']
        observations+=(f" Relative to its epoch-105 initialization, the modified-loss checkpoint changes mean "
                       f"RBC PSNR by {delta_psnr:+.4f} dB and RBC SSIM by {delta_ssim:+.4f}. "
                       "This before/after comparison includes additional training and a fresh optimizer; "
                       "it cannot isolate the loss change from those factors.")
    lines=['# RBC full-split reconstruction comparison','',
        f"Evaluated **{n:,} paired images per method** from the supplied `Validation` split ({manifest['dataset_total']:,} pairs available). "
        + ('This is the complete supplied split.' if n==manifest['dataset_total'] else '**SMOKE TEST ONLY; this is not the full-set result.**'),'',
        '## Methods and checkpoint selection','',
        'The original epoch-105 flow is the exact initialization of the modified-loss model. '
        'The original epoch-95 flow had the highest recorded global SSIM on the 128-image monitoring subset among the saved original checkpoints. '
        'The modified-loss model is the final checkpoint after five additional epochs from epoch 105. '
        'The off-axis method uses the frozen input-only Fourier sideband reconstruction; no target-assisted sign, piston, tilt, or scale fitting enters its prediction. '
        'Raw hologram and constant-0.5 baselines expose background-driven scores.', '',
        '## Findings','',observations,'',
        '## Global metrics','',table(['Method','Images','PSNR ↑','SSIM ↑','MAE ↓','RMSE ↓','LPIPS ↓','FID ↓'],global_rows),'',
        '## RBC and background metrics','',table(['Method','RBC-valid images','RBC PSNR ↑','RBC SSIM ↑','RBC circular RMSE ↓','Background PSNR ↑','Background SSIM ↑'],region_rows),'',
        'RBC masks are fixed target-derived pseudo-regions, not independent anatomical annotations. '
        'Absent regions are excluded only from the corresponding regional mean. Regional SSIM averages the original SSIM map at region centers; windows can include neighboring background.', '',
        '## Borders, circular error, and runtime','',table(['Method','Interior PSNR ↑','Interior SSIM ↑','Global circular RMSE ↓','ms/image','Clipped fraction'],runtime_rows),'',
        'Interior scores remove 16 pixels on each side. Circular error uses the empirically supported proxy of one phase cycle per normalized grayscale interval; radians are not independently calibrated optical phase. '
        'No target-assisted phase-origin alignment is applied. Flow runtime is synchronized GPU batch throughput expressed as milliseconds per image; physics runtime is an individual CPU reconstruction call under concurrent workers. '
        'These timings exclude data loading and metric calculation and are not matched single-image latency measurements.', '',
        f"Physics fallback cases: **{protocol['physics_fallbacks']} / {n}**. Any rejected hologram is explicitly counted and receives the fixed constant-0.5 deployment fallback. Optimizer status for every physics call is retained in the reconstruction log.", '',
        'Data-consistency RMSE is **N/A**: the normalized phase PNG alone does not specify a calibrated common forward operator for the recorded intensity. Unknown reference/amplitude/export parameters prevent a defensible common measurement residual.', '',
        '## Protocol and limits','',
        f"Every flow uses Heun with {manifest['steps']} steps, Gaussian source sigma 1, and seed {manifest['seed']} + 100003 × image index. The same image latent is used across checkpoints and does not depend on batch partition. Model inference uses {manifest['precision']}. The sampler-equivalence check is saved in `sampler_equivalence.json`.", '',
        'All image metrics use predictions clamped to [0,1], fixed data range one, and no independent contrast normalization. PSNR is averaged per image; FID is a single distribution-level score over every real and generated image. '
        'LPIPS uses AlexNet v0.1 with grayscale repeated to RGB in [-1,1]. FID uses 2048-dimensional Inception features and float64 moment accumulation. These natural-image feature metrics are supporting measurements, not validated biological measures.', '',
        'The dataset supplies Training and Validation folders, with no separate untouched test folder. '
        'Training checkpoint monitoring and epoch-95 selection used the first 128 validation images, which are included here as requested. '
        'The prior split audit found training crop siblings for 7,088/7,373 validation files. Consequently these results describe the supplied split and cannot establish independent acquisition-level generalization. '
        'Paired differences are descriptive crop-level comparisons; no independent-sample significance or confidence interval is claimed.', '',
        '## Visual comparison','',
        'The eight illustrated indices are evenly spaced across the sorted evaluation set and were fixed independently of reconstruction quality. All grayscale images share [0,1] display limits.', '',
        *image_links,'','![Metric means and dataset-level FID](metric_comparison.png)','',
        '## Full results and provenance','',
        '- [All aggregate means (CSV)](../metrics/summary.csv)',
        '- [Means, spread, quantiles, and valid counts (JSON)](../metrics/summary.json)',
        '- [Paired per-image differences (CSV)](paired_differences.csv)',
        '- [Metric protocol and package versions](../metrics/metric_protocol.json)',
        '- [Checkpoint hashes, settings, and all paired filenames](../manifest.json)',
        '- [Executed source-file hashes](../source_hashes.txt)',
        *[f'- [{LABELS[name]}: all per-image metrics](../metrics/{name}_per_image.csv)' for name in methods], '',
        'Sources for metric conventions: [LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity), '
        '[TorchMetrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).','']
    (report/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Saved {report / "comparison.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir',required=True)
    main(parser.parse_args())
