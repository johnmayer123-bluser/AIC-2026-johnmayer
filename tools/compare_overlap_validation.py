"""Compare frozen validation predictions on detected-overlap and unmatched strata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def confusion(names, predictions, masks, classes=9, ignore=0):
    total=np.zeros((classes,classes),np.int64); image_scores=[]
    for name in sorted(names):
        with Image.open(masks/(name+'.png')) as im: truth=np.asarray(im)
        with Image.open(predictions/(name+'.png')) as im: pred=np.asarray(im)
        if truth.shape!=pred.shape or truth.ndim!=2 or min(truth.min(),pred.min())<0 or max(truth.max(),pred.max())>=classes:
            raise ValueError('Invalid prediction or mask: '+name)
        keep=truth!=ignore
        table=np.bincount((truth[keep]*classes+pred[keep]).ravel(),minlength=classes**2).reshape(classes,classes)
        total+=table
        inter=np.diag(table);union=table.sum(1)+table.sum(0)-inter
        use=(np.arange(classes)!=ignore)&(union>0)
        image_scores.append(float(np.mean(inter[use]/union[use])))
    inter=np.diag(total);union=total.sum(1)+total.sum(0)-inter
    iou=np.divide(inter,union,out=np.full(classes,np.nan),where=union>0);iou[ignore]=np.nan
    return dict(images=len(names),global_miou=float(np.nanmean(iou)),mean_image_miou=float(np.mean(image_scores)),
                per_class_iou=[None if np.isnan(x) else float(x) for x in iou],valid_pixels=int(total.sum()))


def run(args):
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Refusing to overwrite: '+str(output))
    split_bytes=Path(args.split).read_bytes();split=json.loads(split_bytes)
    train,val=set(split['train']),set(split['val'])
    groups=json.loads(Path(args.groups).read_text(encoding='utf-8'))
    if train&val or any(len(g)!=len(set(g)) for g in groups): raise ValueError('Invalid split or groups')
    leaked={n for group in groups if set(group)&train and set(group)&val for n in group if n in val}
    if not leaked or leaked==val: raise ValueError('Both strata must be nonempty')
    analyses={}
    for item in args.analysis:
        if '=' not in item: raise ValueError('--analysis must be NAME=PREDICTION_DIRECTORY')
        name,path=item.split('=',1)
        if not name or name in analyses: raise ValueError('Empty or duplicate analysis name')
        root=Path(path)
        actual={p.stem for p in root.glob('*.png')}
        if actual!=val: raise ValueError(f'{name} prediction names do not equal split.val')
        analyses[name]=dict(detected_overlap=confusion(leaked,root,Path(args.masks)),
                            no_detected_overlap=confusion(val-leaked,root,Path(args.masks)))
    result=dict(status='complete',split_sha256=hashlib.sha256(split_bytes).hexdigest(),
        groups_sha256=hashlib.sha256(Path(args.groups).read_bytes()).hexdigest(),
        direct_interpretation='Detected-overlap membership is heuristic; strata also differ in scene and class difficulty.',
        validation_images=len(val),detected_overlap_images=len(leaked),no_detected_overlap_images=len(val-leaked),analyses=analyses)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('split','groups','masks','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--analysis',action='append',required=True,help='NAME=PREDICTION_DIRECTORY; repeatable')
    run(p.parse_args())
