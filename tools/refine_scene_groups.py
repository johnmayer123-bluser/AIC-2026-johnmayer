"""Expand an existing RGB overlap audit; isolate strong AND uncertain edges in a draft split."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

try:
    from tools.audit_scenes import components, cv2, geometry, grouped_split, panel, write
except ModuleNotFoundError:
    from audit_scenes import components, cv2, geometry, grouped_split, panel, write


def expanded_pairs(bows, topk):
    if not 1<=topk<len(bows):
        raise ValueError('topk must be in [1, image_count)')
    pairs=set()
    for start in range(0,len(bows),128):
        sims=bows[start:start+128]@bows.T
        for offset,scores in enumerate(sims):
            i=start+offset; scores[i]=-1
            for j in np.argsort(scores,kind='stable')[-topk:]:
                if i!=j:
                    pairs.add(tuple(sorted((i,int(j)))))
    return pairs


def run(args):
    cv2.setNumThreads(1)
    start=time.perf_counter(); source=Path(args.source); out=Path(args.output)
    if out.exists():
        raise FileExistsError('Refusing existing output')
    if json.loads((source/'run.json').read_text())['status']!='complete':
        raise ValueError('Source audit incomplete')
    config=json.loads((source/'config.json').read_text(encoding='utf-8'))
    manifest=Path(config['args']['features']).read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=config['manifest_sha256']:
        raise ValueError('Source manifest changed')
    rows=sorted([r for r in json.loads(manifest) if r['group'] in ('train','val')],key=lambda r:r['name'])
    names=[r['name'] for r in rows]
    out.mkdir(parents=True)
    write(out/'config.json',dict(args=vars(args),source_config=config,
        source_geometry_sha256=hashlib.sha256((source/'geometry.jsonl').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_script_sha256=hashlib.sha256((Path(__file__).parent/'audit_scenes.py').read_bytes()).hexdigest()))
    write(out/'run.json',dict(status='running',stage='load'))
    features=[]
    for n in names:
        with np.load(source/'features'/(n+'.npz'),allow_pickle=False) as f:
            features.append({k:f[k].astype(np.float32) if k=='desc' else f[k] for k in ('xy','desc','gray','counts')})
    previous=[json.loads(line) for line in (source/'geometry.jsonl').read_text().splitlines()]
    done={tuple(r['indices']) for r in previous}
    with np.load(source/'retrieval.npz',allow_pickle=False) as f:
        expanded=expanded_pairs(f['bows'],args.topk)
    new=sorted(expanded-done)
    write(out/'run.json',dict(status='running',stage='geometry',additional_pairs=len(new)))
    results=list(previous)
    def check(pair):
        i,j=pair
        return dict(indices=[i,j],a=names[i],b=names[j],**geometry(features[i],features[j]))
    with (out/'additional_geometry.jsonl').open('w',encoding='utf-8') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in tqdm(pool.map(check,new),total=len(new),desc='expanded geometry',mininterval=5):
            results.append(r); log.write(json.dumps(r,allow_nan=False)+'\n');log.flush()
    strong=[r for r in results if r['status']=='strong_overlap']
    review=[r for r in results if r['status']=='review']
    write(out/'strong_pairs.json',strong);write(out/'review_pairs.json',review)
    strong_groups=components(names,[(r['a'],r['b']) for r in strong])
    guard_groups=components(names,[(r['a'],r['b']) for r in strong+review])
    write(out/'strong_groups.json',strong_groups)
    write(out/'guard_groups.json',[dict(group_id=f'guard{i:04d}',members=g) for i,g in enumerate(guard_groups)])
    split,balance=grouped_split(guard_groups,{n:f['counts'] for n,f in zip(names,features)},args.seed)
    split['note']='Draft guards both strong and uncertain observed edges; false merges and missing overlaps remain possible.'
    write(out/'candidate_split.json',split);write(out/'class_balance.json',balance)
    oldval={r['name'] for r in rows if r['group']=='val'};newval=set(split['val'])
    oldcross=[r for r in strong if (r['a'] in oldval)!=(r['b'] in oldval)]
    initialval=set(json.loads((source/'candidate_split.json').read_text())['val'])
    added_strong=[r for r in strong if tuple(r['indices']) not in done]
    initial_missed_cross=[r for r in added_strong if (r['a'] in initialval)!=(r['b'] in initialval)]
    report=dict(images=len(names),additional_pairs=len(new),total_checked_pairs=len(results),
        status_counts=dict(Counter(r['status'] for r in results)),additional_strong=len(added_strong),
        initial_candidate_newly_found_cross_strong=len(initial_missed_cross),
        strong_groups=len(strong_groups),guard_groups=len(guard_groups),
        guard_non_singleton_groups=sum(len(g)>1 for g in guard_groups),
        guard_largest_groups=[len(g) for g in guard_groups[:20]],
        unknown_singletons=sum(len(g)==1 for g in guard_groups),
        old_cross_strong_pairs=len(oldcross),old_val_with_strong_train_overlap=len({n for r in oldcross for n in (r['a'],r['b']) if n in oldval}),
        candidate_train=len(split['train']),candidate_val=len(split['val']),
        candidate_cross_strong=sum((r['a'] in newval)!=(r['b'] in newval) for r in strong),
        candidate_cross_review=sum((r['a'] in newval)!=(r['b'] in newval) for r in review),
        candidate_val_guard_groups=sum(any(n in newval for n in g) for g in guard_groups),
        old_val_retained=len(newval&oldval),
        limitations=['Top48 is not exhaustive; disconnected groups are not certified distinct scenes.',
            'Guard groups deliberately include uncertain links: conservative false merges are possible.',
            'Draft is not activated; no model score used to select split; no test images used.',
            'Old A00/A01/A02 weights have trained on some draft-val images and cannot evaluate this as unseen data.'])
    write(out/'summary.json',report)
    panels=out/'panels';panels.mkdir()
    chosen=sorted(initial_missed_cross,key=lambda r:-r['inliers'])[:8]+sorted(added_strong,key=lambda r:r['correlation'])[:8]+sorted(review,key=lambda r:-r['inliers'])[:8]
    for r in chosen:
        panel(r,features,names,panels/(r['a']+'_'+r['b']+'.png'))
    # Largest components: evenly spaced filenames to inspect gross false merges, not source proof.
    from PIL import Image, ImageDraw
    lookup={n:i for i,n in enumerate(names)}
    for idx,g in enumerate(guard_groups[:4]):
        chosen=[g[k] for k in np.linspace(0,len(g)-1,min(16,len(g)),dtype=int)]
        canvas=Image.new('RGB',(1024,4*280),'white');draw=ImageDraw.Draw(canvas)
        for k,n in enumerate(chosen):
            x,y=(k%4)*256,(k//4)*280
            draw.text((x+3,y+3),f'{n} group-size={len(g)}',fill='black')
            canvas.paste(Image.fromarray(features[lookup[n]]['gray']).convert('RGB'),(x,y+20))
        canvas.save(panels/f'group_{idx:02d}.png')
    write(out/'run.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start,**report))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--topk',type=int,default=48);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=3407)
    run(p.parse_args())
