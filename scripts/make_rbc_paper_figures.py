"""Create vector comparison charts and a lossless, fixed-scale 8x5 PDF grid.

Consumes completed full-validation metrics and a compact eight-sample export.
No selection by visual quality or independent image contrast normalization.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


METHODS = ['offaxis_input_only','direct_unet_best','scratch_best']
LABELS = ['Physics-informed','Direct U-Net (epoch 138)','Flow (epoch 142)']
SHORT = ['Physics','U-Net','Flow']
PALETTE = ['#4D637A','#CA7039','#398776']
FONT_ALIASES={'Helvetica':'PaperSans','Helvetica-Bold':'PaperSansBold',
              'Times-Roman':'PaperSerif','Times-Bold':'PaperSerifBold'}
# Key, display label, better direction. Every numerical metric in the scorer
# is represented below; mask coverage and clipping are diagnostics.
GROUPS = [
 ('Global fidelity', [
  ('psnr','PSNR (dB)','max'),('ssim','SSIM','max'),('mse','MSE','min'),
  ('rmse','RMSE','min'),('mae','MAE','min'),('circular_rmse_rad','Circular RMSE (rad proxy)','min'),
  ('interior_psnr','Interior PSNR (dB)','max'),('interior_ssim','Interior SSIM','max'),
  ('lpips','LPIPS (AlexNet)','min'),('fid','FID (Inception 2048)','min')]),
 ('RBC pseudo-region fidelity', [
  ('rbc_psnr','RBC PSNR (dB)','max'),('rbc_ssim','RBC SSIM','max'),
  ('rbc_mse','RBC MSE','min'),('rbc_rmse','RBC RMSE','min'),
  ('rbc_circular_rmse_rad','RBC circular RMSE (rad proxy)','min')]),
 ('Background fidelity', [
  ('background_psnr','Background PSNR (dB)','max'),('background_ssim','Background SSIM','max'),
  ('background_mse','Background MSE','min'),('background_rmse','Background RMSE','min'),
  ('background_circular_rmse_rad','Background circular RMSE (rad proxy)','min')]),
 ('Diagnostics', [('rbc_fraction','RBC mask coverage','none'),('clipped_fraction','Prediction clipped fraction','none')])]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def get_mean(summary, method, key):
    return summary[method]['fid'] if key=='fid' else summary[method]['metrics'][key]['mean']


def text(c,x,y,value,size=9,font='Helvetica',color='#20252B',align='left'):
    c.setFillColor(colors.HexColor(color));c.setFont(FONT_ALIASES[font],size)
    {'left':c.drawString,'right':c.drawRightString,'center':c.drawCentredString}[align](x,y,str(value))


def fmt(value,key=None):
    if value is None or not math.isfinite(float(value)): return 'N/A'
    if key=='fid' or (key and ('psnr' in key or 'timing' in key or key=='runtime_ms')):
        return f'{value:.3f}'
    return f'{value:.5f}'


def grid(root,out):
    arrays=np.load(root/'paper_samples.npz')
    meta=read_json(root/'paper_samples.json')
    images=arrays['images']
    if images.shape != (8,5,256,256) or not np.isfinite(images).all():
        raise ValueError('Expected eight finite native-resolution samples and five aligned columns.')
    if arrays['indices'].tolist()!=meta['indices']: raise ValueError('Sample metadata mismatch.')
    width=7.1*72; margin=8; labels_width=16; gap=3
    tile=(width-2*margin-labels_width-4*gap)/5
    top=27; bottom=30; height=top+8*tile+7*gap+bottom
    path=out/'rbc_eight_sample_grid.pdf'
    c=canvas.Canvas(str(path),pagesize=(width,height),pageCompression=1,initialFontName='PaperSans')
    c.setTitle('RBC reconstruction: eight fixed validation samples')
    c.setAuthor('RBC reconstruction comparison')
    columns=['Input','Ground truth','Physics-informed','Direct U-Net','Flow']
    for col,title in enumerate(columns):
        x=margin+labels_width+col*(tile+gap)
        text(c,x+tile/2,height-15,title,size=9.3,font='Times-Bold',align='center',color='#111111')
    for row in range(8):
        y=height-top-(row+1)*tile-row*gap
        text(c,margin,y+tile/2-3,f'({chr(97+row)})',size=9,font='Times-Roman',color='#111111')
        for col in range(5):
            x=margin+labels_width+col*(tile+gap)
            pixel=(np.clip(images[row,col],0,1)*255).round().astype(np.uint8)
            c.drawImage(ImageReader(Image.fromarray(pixel)),x,y,width=tile,height=tile,mask=None)
    bar_width=96;bar_x=(width-bar_width)/2
    for j in range(128):
        c.setFillColorRGB(j/127,j/127,j/127)
        c.rect(bar_x+j*bar_width/128,13,bar_width/128+.1,5,fill=1,stroke=0)
    text(c,bar_x-4,12,'0',size=7,align='right')
    text(c,bar_x+bar_width+4,12,'1',size=7)
    text(c,width/2,3,'Shared normalized grayscale range',size=7,font='Times-Roman',align='center')
    c.save()
    return path


def legend(c,x,y):
    for i,(name,color) in enumerate(zip(SHORT,PALETTE)):
        xx=x+i*91
        c.setFillColor(colors.HexColor(color));c.rect(xx,y-1,10,7,fill=1,stroke=0)
        text(c,xx+15,y,name,size=9)


def chart(c,x,y,w,h,label,values,direction,log=False):
    suffix='higher is better' if direction=='max' else 'lower is better' if direction=='min' else 'diagnostic'
    lines=['']
    for word in label.split():
        candidate=(lines[-1]+' '+word).strip()
        if lines[-1] and stringWidth(candidate,'PaperSansBold',9)>w-16: lines.append(word)
        else: lines[-1]=candidate
    for i,line in enumerate(lines): text(c,x+8,y+h-16-i*11,line,size=9,font='Helvetica-Bold')
    text(c,x+8,y+h-28-(len(lines)-1)*11,suffix+(' | log scale' if log else ''),size=7.2,color='#606870')
    left=x+28; right=x+w-8; bottom=y+23; top=y+h-49-(len(lines)-1)*11
    finite=[v for v in values if v is not None and math.isfinite(v)]
    if not finite: return
    if log:
        lo=math.floor(math.log10(min(finite)))-.2; hi=math.ceil(math.log10(max(finite)))+.2
        transform=lambda v:bottom+(math.log10(v)-lo)/(hi-lo)*(top-bottom)
        ticks=[10**k for k in range(math.ceil(lo),math.floor(hi)+1)]
    else:
        upper=max(finite)*1.22 if max(finite)>0 else 1
        transform=lambda v:bottom+v/upper*(top-bottom)
        ticks=[0,upper/2,upper]
    c.setStrokeColor(colors.HexColor('#E0E5E9'));c.setLineWidth(.4)
    for tick in ticks:
        yy=transform(tick)
        c.line(left,yy,right,yy)
        text(c,left-4,yy-2,f'{tick:.2g}',size=6.8,color='#6A7279',align='right')
    for i,(v,color) in enumerate(zip(values,PALETTE)):
        center=left+(i+.5)*(right-left)/3; bw=(right-left)/5.2
        if v is None: continue
        yy=transform(v)
        c.setFillColor(colors.HexColor(color));c.rect(center-bw/2,bottom,bw,max(0,yy-bottom),fill=1,stroke=0)
        value=f'{v:.3f}' if abs(v)>=1 else f'{v:.4f}'
        text(c,center,yy+4,value,size=8,font='Helvetica-Bold',align='center')
        text(c,center,bottom-12,SHORT[i],size=8,align='center')


def comparison(root,out,summary,timing):
    path=out/'rbc_all_metrics_comparison.pdf'
    page_w,page_h=landscape(A4)
    c=canvas.Canvas(str(path),pagesize=(page_w,page_h),pageCompression=1,initialFontName='PaperSans')
    c.setTitle('RBC full-validation metric and runtime comparison')
    n=summary[METHODS[0]]['samples'];valid=summary[METHODS[1]]['metrics']['rbc_psnr']['n']
    text(c,30,page_h-29,'RBC reconstruction: full validation comparison',size=16,font='Helvetica-Bold')
    text(c,30,page_h-46,f'{n:,} paired images per method | {valid:,} valid RBC pseudo-regions | means of per-image scores; FID is dataset-level',size=9,color='#52606B')
    xs=[30,422,610,798]
    y=page_h-68
    for x,label in zip(xs[1:],LABELS): text(c,x,y,label,size=10,font='Helvetica-Bold',align='right')
    y-=8
    c.setStrokeColor(colors.HexColor('#34424D'));c.line(30,y,page_w-30,y)
    rows=[]
    for group,items in GROUPS:
        rows.append((group,None,None,'group'))
        for key,label,direction in items:
            rows.append((label,[get_mean(summary,m,key) for m in METHODS],key,direction))
    rows.append(('Reconstruction time',None,None,'group'))
    rows.append(('Single-image latency, mean (ms)',[timing['single_image'][m]['mean_ms'] for m in METHODS],'timing','min'))
    rows.append(('Single-image latency, median (ms)',[timing['single_image'][m]['median_ms'] for m in METHODS],'timing','min'))
    rows.append(('Single-image latency, 95th percentile (ms)',[timing['single_image'][m]['p95_ms'] for m in METHODS],'timing','min'))
    rows.append(('Batch-16 amortized GPU time (ms/image)',[None]+[timing['batch16_amortized'][m]['mean_ms'] for m in METHODS[1:]],'timing','min'))
    rows.append(('Full-set cached-pass timing (ms/image)*',[get_mean(summary,m,'runtime_ms') for m in METHODS],'runtime_ms','none'))
    rows.append(('Data-consistency RMSE (uncalibrated operator)',[None,None,None],'dc_rmse','none'))
    row_h=13.0
    for k,(label,values,key,direction) in enumerate(rows):
        y-=row_h
        if direction=='group':
            c.setFillColor(colors.HexColor('#E8EDF1'));c.rect(30,y-3,page_w-60,row_h,fill=1,stroke=0)
            text(c,35,y,label,size=8.2,font='Helvetica-Bold')
            continue
        text(c,35,y,label+(' (+)' if direction=='max' else ' (-)' if direction=='min' else ''),size=8.2)
        finite=[v for v in values if v is not None]
        best=max(finite) if finite and direction=='max' else min(finite) if finite and direction=='min' else None
        for x,value in zip(xs[1:],values):
            text(c,x,y,fmt(value,key),size=8.4,font='Helvetica-Bold' if best is not None and value==best else 'Helvetica',align='right')
    foot=[
      '(+) higher is better; (-) lower is better. Bold marks the best mean among applicable methods; no significance claim.',
      'Fresh timing: 128 fixed, evenly spaced images, serial execution after warmup. Physics uses CPU; both networks use the same GPU.',
      '* Historical full-pass timing: neural batch-16 throughput; physics call time under four concurrent CPU workers. These are not single-image latencies.',
      'Times exclude image loading and metric calculation. Phase/circular errors use PNG proxies; masks are heuristic. No calibrated data-consistency score is available.',
      'Training/validation share parent-image groups; the validation set includes the 128 checkpoint-selection images. Results concern this supplied split.'
    ]
    for i,line in enumerate(foot): text(c,30,53-i*9,line,size=7.1,color='#505A63')
    c.showPage()
    panels=[(key,label,direction) for group,items in GROUPS for key,label,direction in items]
    # Two pages cover all fidelity metrics; diagnostics and runtimes have a dedicated page.
    pages=[panels[:10],panels[10:20],panels[20:]+[
        ('latency','Single-image latency (ms)','min'),('batch16','Batch-16 throughput (ms/image)','min'),
        ('runtime_ms','Full-set pass time (ms/image)','none')]]
    for page,items in enumerate(pages,start=2):
        title=['Global fidelity and perceptual metrics','RBC and background fidelity','Diagnostics and reconstruction time'][page-2]
        text(c,30,page_h-29,title,size=16,font='Helvetica-Bold');legend(c,30,page_h-47)
        columns=5 if len(items)==10 else 3; rows=math.ceil(len(items)/columns)
        panel_w=(page_w-56)/columns;panel_h=(page_h-108)/max(rows,2)
        for j,(key,label,direction) in enumerate(items):
            x=28+(j%columns)*panel_w;y=page_h-69-(j//columns+1)*panel_h
            values=([timing['single_image'][m]['mean_ms'] for m in METHODS] if key=='latency' else
                    [None]+[timing['batch16_amortized'][m]['mean_ms'] for m in METHODS[1:]] if key=='batch16' else
                    [get_mean(summary,m,key) for m in METHODS])
            chart(c,x,y,panel_w,panel_h,label,values,direction,log=key in ['latency','batch16','runtime_ms'])
        text(c,30,27,f'Full quality metrics: {n:,} images; RBC fidelity: {valid:,} valid masks. Fresh timings: 128 images. See page 1 for definitions and limitations.',size=8,color='#505A63')
        text(c,page_w-30,13,str(page),size=8,align='right');c.showPage()
    c.save()
    return path


def main(args):
    font_dir=Path(args.font_dir)
    for name,file in [('PaperSans','arial.ttf'),('PaperSansBold','arialbd.ttf'),
                       ('PaperSerif','times.ttf'),('PaperSerifBold','timesbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    root=Path(args.result_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    summary=read_json(root/'metrics/summary.json');completed=read_json(root/'metrics/complete.json')
    timing=read_json(root/'matched_timing.json');meta=read_json(root/'paper_samples.json')
    if completed['status']!='complete' or completed['samples_per_method']!=7373:
        raise ValueError('Paper figures require a completed full validation evaluation.')
    if any(summary[m]['samples']!=7373 or summary[m]['fid_real_samples']!=7373 or summary[m]['fid_fake_samples']!=7373 for m in METHODS):
        raise ValueError('Incomplete per-method metric coverage.')
    expected=set(summary[METHODS[0]]['metrics'])|{'fid'}
    displayed={key for _,items in GROUPS for key,_,_ in items}|{'runtime_ms'}
    if expected!=displayed: raise ValueError(f'Metric coverage mismatch: {expected^displayed}')
    paths=[grid(root,out),comparison(root,out,summary,timing)]
    rows=[]
    for _,items in GROUPS:
        for key,label,direction in items:
            for method in METHODS:
                rows.append({'metric':key,'label':label,'method':method,'better':direction,
                    'mean':get_mean(summary,method,key),'valid_n':7373 if key=='fid' else summary[method]['metrics'][key]['n']})
    for key,source in [('single_image_ms','single_image'),('batch16_ms_per_image','batch16_amortized')]:
        for method in METHODS:
            item=timing[source].get(method)
            rows.append({'metric':key,'label':key,'method':method,'better':'min',
                         'mean':item['mean_ms'] if item else None,'valid_n':item['n'] if item else 0})
    with (out/'rbc_comparison_values.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    caption=("Eight evenly spaced examples from the supplied RBC validation split. Columns show the input hologram, paired ground-truth phase PNG, "
        "input-only off-axis Fourier reconstruction, direct supervised U-Net (best epoch 138), and RBC-focused conditional flow model (best epoch 142 of 200; Heun, 40 steps). "
        "All panels use the same normalized grayscale range [0,1] without individual contrast adjustment. Rows (a)-(h) correspond to zero-based validation indices "
        +', '.join(map(str,meta['indices']))+". Examples were selected independently of reconstruction quality. The phase images are PNG-encoded proxies; no physical scale bar is available.\n")
    (out/'rbc_figure_caption.txt').write_text(caption,encoding='utf-8')
    (out/'figure_provenance.json').write_text(json.dumps({'result_dir':str(root.resolve()),'samples':meta,
        'pdfs':[str(p.resolve()) for p in paths],'raster_embedding':'native 256x256 lossless grayscale; vector text and vector charts; embedded TrueType fonts',
        'normalization':'shared fixed [0,1], display clamp only'},indent=2))
    print('\n'.join(str(p.resolve()) for p in paths))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir',required=True)
    parser.add_argument('--out_dir',default='output/pdf')
    parser.add_argument('--font_dir',default='C:/Windows/Fonts',help='Directory with Arial and Times New Roman TTF files for embedded paper fonts.')
    main(parser.parse_args())
