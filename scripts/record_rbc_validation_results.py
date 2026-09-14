"""Verify a completed best-checkpoint evaluation and record its full results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_result(root):
    root=Path(root)
    manifest=read_json(root/'manifest.json')
    complete=read_json(root/'metrics/complete.json')
    summaries=read_json(root/'metrics/summary.json')
    n=manifest['samples']
    if n != manifest['dataset_total'] or complete['status'] != 'complete' or complete['samples_per_method'] != n:
        raise ValueError('A completed full-split evaluation is required.')
    all_rows={}
    for method in complete['methods']:
        item=summaries[method]
        if (item['samples'], item['fid_real_samples'], item['fid_fake_samples']) != (n,n,n) or not math.isfinite(item['fid']):
            raise ValueError(f'Incomplete metrics or FID: {method}')
        with (root/'metrics'/f'{method}_per_image.csv').open(encoding='utf-8') as handle:
            rows=list(csv.DictReader(handle))
        if [int(row['index']) for row in rows] != list(range(n)):
            raise ValueError(f'Missing, duplicate, or unordered rows: {method}')
        if any(row['hologram'] != pair['hologram'] or row['target'] != pair['target']
               for row,pair in zip(rows,manifest['paths'])):
            raise ValueError(f'Image pairing mismatch: {method}')
        for key,stat in item['metrics'].items():
            values=np.array([float(row[key]) for row in rows if row[key] != ''])
            if not np.isfinite(values).all() or len(values) != stat['n']:
                raise ValueError(f'Invalid value/count: {method}, {key}')
            if len(values) and not math.isclose(float(values.mean()),stat['mean'],rel_tol=1e-10,abs_tol=1e-9):
                raise ValueError(f'Aggregate mismatch: {method}, {key}')
        all_rows[method]=rows
    return manifest,summaries,all_rows


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     +['| '+' | '.join(map(str,row))+' |' for row in rows])


def main(args):
    root=Path(args.result_dir).resolve()
    reference=Path(args.reference_dir).resolve()
    output=Path(args.output_md).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest,summary,rows=validate_result(root)
    old_manifest,old_summary,old_rows=validate_result(reference)
    for key in ['paths','samples','dataset_total','seed','seed_stride','solver','steps','precision','batch_size']:
        if manifest[key] != old_manifest[key]:
            raise ValueError(f'Historical benchmark protocol mismatch: {key}')
    def hashes(path):
        return {line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0]
                for line in path.read_text().splitlines() if line.strip()}
    new_hashes,old_hashes=hashes(root/'source_hashes.txt'),hashes(reference/'source_hashes.txt')
    compared_sources=['scripts/benchmark_rbc_reconstruction.py','scripts/score_rbc_benchmark.py',
                      'lensless_flow/sampler.py','lensless_flow/metrics.py','lensless_flow/rbc_regions.py']
    for name in compared_sources:
        if new_hashes[name] != old_hashes[name]:
            raise ValueError(f'Executed benchmark source changed: {name}')
    selection=read_json(root/'checkpoint_selection.json')
    candidate=args.candidate
    if manifest['models'][candidate]['sha256'] != selection['sha256']:
        raise ValueError('Best-checkpoint hash mismatch.')
    selected=summary[candidate]
    metrics=selected['metrics']
    n=manifest['samples']
    references=['original_e105','original_e95','region_final','offaxis_input_only','measurement','constant_0_5']
    combined={candidate:selected,**{name:old_summary[name] for name in references}}
    labels={candidate:f'Scratch modified loss, best epoch {selection["best_epoch"]}',
            'original_e105':'Original flow, epoch 105','original_e95':'Original flow, epoch 95',
            'region_final':'Modified-loss five-epoch fine-tune','offaxis_input_only':'Off-axis physics',
            'measurement':'Raw hologram','constant_0_5':'Constant 0.5'}
    excluded=[int(row['index']) for row in rows[candidate] if row['rbc_psnr']=='']
    old_excluded=[int(row['index']) for row in old_rows['region_final'] if row['rbc_psnr']=='']
    if excluded != old_excluded:
        raise ValueError('Regional valid-image set changed.')
    paired=[]
    for base in references:
        for key,higher in [('psnr',True),('ssim',True),('rbc_psnr',True),('rbc_ssim',True),
                           ('lpips',False),('rbc_mse',False),('background_mse',False),('circular_rmse_rad',False)]:
            values=np.array([float(c[key])-float(r[key]) for c,r in zip(rows[candidate],old_rows[base])
                             if c[key] != '' and r[key] != ''])
            paired.append(dict(candidate=candidate,reference=base,metric=key,n=len(values),
                mean_difference=float(values.mean()),median_difference=float(np.median(values)),
                win_fraction=float((values>0 if higher else values<0).mean())))
    report=root/'report'
    report.mkdir(exist_ok=True)
    with (report/'historical_paired_differences.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    (report/'historical_comparison.json').write_text(json.dumps(combined,indent=2,allow_nan=False),encoding='utf-8')
    verification=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='complete',
        samples=n,best_epoch=selection['best_epoch'],checkpoint_sha256=selection['sha256'],
        per_image_order_and_pairing='verified',all_metric_means_and_counts='verified',
        fid_real_samples=selected['fid_real_samples'],fid_fake_samples=selected['fid_fake_samples'],
        historical_protocol_and_sources='identical on compared settings and inference/metric source files',
        rbc_valid=metrics['rbc_psnr']['n'],rbc_excluded_indices=excluded)
    (report/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    def link(path):
        return os.path.relpath(path,output.parent).replace('\\','/')
    def value(v):
        return 'N/A' if v is None else f'{v:.6f}'
    comparison=[]
    for name,item in combined.items():
        comparison.append([labels[name],*[value(item['metrics'][key]['mean']) for key in
                           ['psnr','ssim','rbc_psnr','rbc_ssim','lpips']],value(item['fid'])])
    all_metrics=[[key,stat['n'],value(stat['mean']),value(stat['std']),value(stat['median'])]
                 for key,stat in metrics.items()]
    changes=[]
    for base in ['original_e95','region_final','offaxis_input_only']:
        old=old_summary[base]
        changes.append([labels[base],*[f"{metrics[k]['mean']-old['metrics'][k]['mean']:+.6f}" for k in
                       ['psnr','ssim','rbc_psnr','rbc_ssim','lpips']],f"{selected['fid']-old['fid']:+.6f}"])
    # Separate the already monitored subset from the rest; FID is only full-set.
    subsets=[]
    for title,part in [('First 128 monitoring images',rows[candidate][:128]),('Remaining validation images',rows[candidate][128:])]:
        subsets.append([title,len(part),*[value(float(np.mean([float(r[k]) for r in part if r[k] != ''])))
                         for k in ['psnr','ssim','rbc_psnr','rbc_ssim']]])
    lines=['# Best from-scratch RBC flow: full validation results','',
        f"Evaluation and result verification completed at {verification['verified_utc']}. "
        f"The model trained for **{selection['completed_epoch']} epochs**. Its selected checkpoint is **epoch {selection['best_epoch']}**, "
        f"chosen by highest RBC SSIM on the fixed {selection['selection_samples']}-image monitoring subset. "
        f"This evaluation covers **all {n:,} validation pairs**; no new checkpoint or loss weight was chosen from these full-set results.", '',
        '## Main comparison','',
        markdown_table(['Method','Global PSNR ↑','Global SSIM ↑','RBC PSNR ↑','RBC SSIM ↑','LPIPS ↓','FID ↓'],comparison),'',
        'The historical rows are copied from the completed earlier full-set benchmark. Ordered image pairs, sampling settings, precision, inference code, and metric code were checked for consistency. '
        'PSNR is in dB; image-level scores are averaged per image. FID is computed once from the complete real and predicted distributions.', '',
        '## Change relative to previous methods','',
        'Entries below are new minus reference. Positive PSNR/SSIM and negative LPIPS/FID indicate improvement.', '',
        markdown_table(['Reference','Δ global PSNR','Δ global SSIM','Δ RBC PSNR','Δ RBC SSIM','Δ LPIPS','Δ FID'],changes),'',
        f"RBC-only scores use **{metrics['rbc_psnr']['n']:,} valid pseudo-regions**. Excluded indices: {', '.join(map(str,excluded))}. "
        'These images remain in all global scores and FID. The excluded set matches the earlier evaluation. '
        'Mean improvements need not imply that most images improve; paired differences and win fractions are preserved in the linked CSV.', '',
        '## Every recorded metric for the selected checkpoint','',
        markdown_table(['Metric','Valid images','Mean','Sample standard deviation','Median'],all_metrics),'',
        f"FID: **{selected['fid']:.6f}**, with **{selected['fid_real_samples']:,} real and {selected['fid_fake_samples']:,} predicted images**. "
        'Data-consistency RMSE is N/A because a calibrated common intensity forward operator is unavailable.', '',
        '## Monitoring subset and remaining validation images','',
        markdown_table(['Subset','Images','Global PSNR','Global SSIM','RBC PSNR','RBC SSIM'],subsets),'',
        'The first 128 images were used for checkpoint selection and are included in the requested full validation set. '
        'The remaining rows were not used by this run for epoch selection, but the known overlap of parent-image groups between training and validation still limits independence.', '',
        '## Reproduction and limits','',
        f"The checkpoint SHA256 is `{selection['sha256']}`. All reconstructions use 40-step Heun, Gaussian source standard deviation 1, "
        'image seed `20260903 + 100003 × index`, batch size 16, and float32 with TF32 enabled. Targets never enter inference. '
        'Metrics clamp predictions to [0,1] and use a fixed range of one, without per-image contrast normalization or target-assisted phase alignment.', '',
        'RBC/background regions are frozen target-derived pseudo-masks. Regional SSIM selects centers from the ordinary SSIM map; its windows can cross a region boundary. '
        'Circular errors use a 2π-per-grayscale-interval proxy, not independently calibrated optical radians. '
        'LPIPS uses AlexNet v0.1 and FID uses 2048-dimensional Inception features, with grayscale repeated to RGB. '
        'They are supporting image metrics rather than validated measures of cellular biology. Runtime excludes data loading and metrics.', '',
        'This is the supplied Validation split, not an independent test dataset. The previous audit found crop siblings in training for 7,088/7,373 validation files. '
        'Comparisons between training runs also differ in training duration, initialization, and LR scheduling, so they do not isolate a causal effect of the modified loss.', '',
        '## Saved artifacts','',
        f"- [Complete generated report and image grids]({link(report/'comparison.md')})",
        f"- [All per-image scores]({link(root/'metrics'/f'{candidate}_per_image.csv')})",
        f"- [Full metric statistics and FID]({link(root/'metrics'/f'{candidate}_summary.json')})",
        f"- [Paired changes against earlier methods]({link(report/'historical_paired_differences.csv')})",
        f"- [Verification record]({link(report/'verification.json')})",
        f"- [Checkpoint selection and hash]({link(root/'checkpoint_selection.json')})",
        f"- [Metric definitions and versions]({link(root/'metrics/metric_protocol.json')})",
        f"- [Training run]({selection['wandb_training_run']})",'']
    output.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'recorded','samples':n,'best_epoch':selection['best_epoch'],'report':str(output)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir',required=True)
    parser.add_argument('--reference_dir',required=True)
    parser.add_argument('--output_md',required=True)
    parser.add_argument('--candidate',default='scratch_best')
    main(parser.parse_args())
