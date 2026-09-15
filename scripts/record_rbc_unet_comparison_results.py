"""Validate all per-image records and write the direct/physics/flow report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.record_rbc_validation_results import validate_result, markdown_table


def main(args):
    root=Path(args.result_dir).resolve()
    manifest,summary,rows=validate_result(root)
    selection=json.loads((root/'checkpoint_selection.json').read_text())
    timing=json.loads((root/'matched_timing.json').read_text())
    provenance=json.loads((root/'cache_reproduction_check.json').read_text())
    if manifest['models']['direct_unet_best']['sha256']!=selection['sha256']:
        raise ValueError('Selected U-Net checkpoint does not match the benchmark.')
    methods=['offaxis_input_only','direct_unet_best','scratch_best']
    labels=['Physics-informed off-axis','Direct U-Net, epoch 138','RBC-focused flow, epoch 142']
    def value(method,key):
        return summary[method]['metrics'][key]['mean']
    keys=list(summary[methods[0]]['metrics'])
    comparison=[]
    for key in keys:
        if any(key not in summary[m]['metrics'] for m in methods): raise ValueError(key)
        comparison.append([key,*['N/A' if value(m,key) is None else f'{value(m,key):.6f}' for m in methods],
                           '/'.join(str(summary[m]['metrics'][key]['n']) for m in methods)])
    comparison.append(['FID',*[f'{summary[m]["fid"]:.6f}' for m in methods],'7373/7373/7373'])
    speed=[]
    for m,label in zip(methods,labels):
        t=timing['single_image'][m];batch=timing['batch16_amortized'].get(m)
        speed.append([label,t['n'],f'{t["mean_ms"]:.3f}',f'{t["median_ms"]:.3f}',f'{t["p95_ms"]:.3f}',
                      f'{batch["mean_ms"]:.3f}' if batch else 'N/A'])
    paired=[]
    for reference in ['scratch_best','offaxis_input_only']:
        for key in ['psnr','ssim','rbc_psnr','rbc_ssim','lpips','mse','circular_rmse_rad']:
            differences=[float(a[key])-float(b[key]) for a,b in zip(rows['direct_unet_best'],rows[reference]) if a[key] and b[key]]
            paired.append({'reference':reference,'metric':key,'n':len(differences),'mean_delta_unet_minus_reference':sum(differences)/len(differences)})
    (root/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    n=manifest['samples'];valid=summary['direct_unet_best']['metrics']['rbc_psnr']['n']
    delta_psnr=value('direct_unet_best','psnr')-value('scratch_best','psnr')
    delta_rbc=value('direct_unet_best','rbc_ssim')-value('scratch_best','rbc_ssim')
    lines=[
      '# Direct U-Net, physics-informed reconstruction, and flow: full validation comparison','',
      '**15 September 2026.** U-Net training was stopped at the user\'s request during epoch 139, after 138 completed epochs. '
      'The interrupted partial epoch was not evaluated or saved. The selected checkpoint is epoch 138, the best saved model by both RBC SSIM and global SSIM on the fixed 128-image monitoring subset. '
      'The checkpoint contains 508,668 completed optimizer updates. It was configured for 200 epochs but did not complete that budget.','',
      f'All **{n:,} validation pairs** were evaluated with the direct U-Net. The comparison uses the completed 200-epoch modified-loss flow model\'s best checkpoint (epoch 142), and the frozen input-only off-axis physics method. '
      f'RBC metrics cover **{valid:,} valid target-derived pseudo-regions**; other full-image scores use all {n:,} images.','',
      '## Findings','',
      f'The direct U-Net improves global PSNR over the selected flow model by {delta_psnr:.4f} dB and RBC SSIM by {delta_rbc:.6f}. '
      'It has the highest global and RBC PSNR/SSIM among these three methods. Its LPIPS is better than the flow model\'s, while the physics method has the lowest LPIPS. '
      f'The U-Net FID is {summary["direct_unet_best"]["fid"]:.4f}, versus {summary["scratch_best"]["fid"]:.4f} for flow and {summary["offaxis_input_only"]["fid"]:.4f} for physics. '
      'Thus the U-Net does not win across all criteria. FID and LPIPS use natural-image feature networks and are supporting metrics rather than calibrated biological fidelity measures.','',
      '## All quantitative metrics','',
      'Arithmetic means of per-image metrics are shown, except FID, which is computed once over the complete paired distributions. '
      'Each method has 7,373 real and 7,373 predicted images in FID. Counts are ordered physics / U-Net / flow.','',
      markdown_table(['Metric',*labels,'Valid counts'],comparison),'',
      '`runtime_ms` above comes from each method\'s full-set reconstruction pass: synchronized neural GPU batch-16 wall time divided by the actual batch size, versus individual physics CPU call time with concurrent workers. '
      'These are historical throughput/call measurements; the matched fresh timing experiment below supplies the primary latency comparison.','',
      '## Fresh matched timing','',
      markdown_table(['Method','Timed images','Mean latency (ms)','Median (ms)','P95 (ms)','Batch-16 GPU ms/image'],speed),'',
      timing['protocol'],'',
      f'Hardware: {timing["hardware"]["gpu"]}; CPU: {timing["hardware"]["cpu"]}; Torch CPU threads: {timing["hardware"]["torch_cpu_threads"]}. '
      'The same 128 evenly spaced validation images were used for every method, with no concurrent reconstruction jobs. '
      'Timing is reconstruction time per sample, not the time to compute PSNR/SSIM/LPIPS/FID. FID is inherently a set-level evaluation. '
      'Original per-image timings and mean/median/P05/P95 values are saved in `matched_timing.json`.','',
      '## Provenance and evaluation protocol','',
      'The new U-Net predictions were generated on the complete validation split in float32 with TF32 enabled, batch size 16, with one deterministic network evaluation per image. '
      'The flow and physics full-set prediction caches and metric records were reused from the previously completed validation evaluation. '
      'Before reuse, the script verified image pairing and order, metric/inference source files, the RBC-mask function AST, package versions, and GPU identity. '
      'A fresh flow reconstruction of the first matching batch and fresh physics reconstructions at eight evenly spaced indices were compared with the cached arrays.','',
      f'Flow cache reproduction maximum absolute discrepancy: `{provenance["flow_first_batch_max_abs_difference"]}`. '
      f'Physics cache reproduction maximum discrepancy: `{max(provenance["physics_eight_evenly_spaced_max_abs_differences"])}`. '
      'File and checkpoint hashes, source paths, method settings, and all paired filenames are retained in the manifest and provenance JSON files.','',
      'Every method is scored with [0,1] clipping, fixed data range one, and no independent contrast normalization. '
      'Regional SSIM uses the existing full-image SSIM map with centers restricted to the mask. Circular errors assume one 2-pi cycle per normalized grayscale interval. '
      'Interior metrics remove 16 pixels from each edge. RBC masks are fixed target-derived pseudo-regions; six invalid masks are excluded only from their corresponding RBC means. '
      'No target-assisted phase origin, sign, or tilt fitting enters the physical reconstruction. Data-consistency RMSE is unavailable because no calibrated common intensity forward operator is supplied.','',
      'The validation set contains the monitoring images used for checkpoint selection. The prior audit found training crop siblings for 7,088 of the 7,373 validation files. '
      'These results describe the provided split, not an untouched acquisition-level test set. The U-Net was stopped early; this comparison does not isolate architecture, training duration, and loss as separate causal factors.','',
      '## Paper figures','',
      'The single-page grid contains eight rows and five columns: input, ground truth, physics-informed, direct U-Net, and flow. '
      'The eight sorted validation indices are evenly spaced and fixed independently of image quality. All images retain native 256x256 samples and use one grayscale range. '
      'Text and charts are vector PDF elements; image tiles use lossless grayscale embedding. No physical scale bar is invented.','',
      '- `output/pdf/rbc_eight_sample_grid.pdf`','- `output/pdf/rbc_all_metrics_comparison.pdf`',
      '- `output/pdf/rbc_figure_caption.txt`','- `output/pdf/rbc_comparison_values.csv`','',
      'Full per-image CSV records, aggregate means/spread/quantiles, timing observations, and sample identities are in `outputs/rbc_unet_best_comparison/results/`.','']
    output=Path(args.output_md);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')
    verification={'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'complete',
      'samples_per_method':n,'rbc_valid':valid,'unet_best_epoch':selection['best_epoch'],
      'checkpoint_sha256':selection['sha256'],'all_per_image_indices_and_pairs':'verified',
      'all_metric_means_and_valid_counts':'recomputed and verified','fid_real_and_fake_samples':n,
      'timing_samples_per_method':len(timing['indices'])}
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2));print(output.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir',required=True)
    parser.add_argument('--output_md',default='docs/rbc_unet_best_validation_comparison.md')
    main(parser.parse_args())
