"""Conservative overlap graph; candidate split only, never a scene-independence certificate.

Requires optional opencv-python-headless. No neural model, test data or extra data.
Stages are restartable from checked input/script hashes; original files stay untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def components(names, edges):
    parent = {n: n for n in names}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for a, b in edges:
        x, y = find(a), find(b)
        if x != y:
            parent[max(x,y)] = min(x,y)
    groups = defaultdict(list)
    for n in sorted(names):
        groups[find(n)].append(n)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def extract(task):
    row, mask_root, folder = task
    destination = folder / (row['name'] + '.npz')
    raw = Path(row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
        raise ValueError('Image changed since RGB audit: ' + row['name'])
    if destination.exists():
        return
    with Image.open(io.BytesIO(raw)) as im:
        metadata = dict(info={str(k): str(v)[:1000] for k,v in im.info.items()},
                        exif={str(k): str(v)[:1000] for k,v in im.getexif().items()})
        gray = np.asarray(im.convert('L').resize((512,512), Image.Resampling.BILINEAR))
    kp, desc = cv2.SIFT_create(nfeatures=256).detectAndCompute(gray, None)
    xy = np.array([p.pt for p in kp], np.float32).reshape(-1,2)
    desc = desc.astype(np.uint8) if desc is not None else np.empty((0,128), np.uint8)
    with Image.open(mask_root / (row['name']+'.png')) as im:
        mask = np.asarray(im)
        if mask.ndim != 2 or mask.shape != (1024,1024) or mask.max() > 8:
            raise ValueError('Invalid mask: '+row['name'])
        counts = np.bincount(mask.ravel(), minlength=9)
    np.savez_compressed(destination, xy=xy, desc=desc, gray=cv2.resize(gray,(256,256)),
                        counts=counts, metadata=json.dumps(metadata, ensure_ascii=False))


def retrieve(features, rows, seed, topk):
    rng = np.random.default_rng(seed)
    sample = np.concatenate([f['desc'][rng.choice(len(f['desc']), min(12,len(f['desc'])), replace=False)]
                             for f in features if len(f['desc'])]).astype(np.float32)
    if len(sample) > 50000:
        sample = sample[rng.choice(len(sample),50000,replace=False)]
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(sample, min(256,len(sample)), None,
                              (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,.1),1,cv2.KMEANS_PP_CENTERS)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    matcher.add([centers]); matcher.train()
    bows = []
    for f in tqdm(features, desc='visual word retrieval', mininterval=5):
        matches = matcher.match(f['desc']) if len(f['desc']) else []
        bows.append(np.bincount([m.trainIdx for m in matches],minlength=len(centers)))
    bows = np.asarray(bows, np.float32)
    idf = np.log((1+len(bows))/(1+(bows>0).sum(0)))+1
    bows = np.sqrt(bows)*idf
    bows /= np.maximum(np.linalg.norm(bows,axis=1,keepdims=True),1e-8)
    pairs = set()
    popcount = np.array([i.bit_count() for i in range(256)],np.uint8)
    hashes = {k: np.array([list(bytes.fromhex(r[k])) for r in rows],np.uint8) for k in ('phash','dhash')}
    for start in range(0,len(rows),128):
        sim = bows[start:start+128] @ bows.T
        for j, scores in enumerate(sim):
            i = start+j
            scores[i] = -1
            neighbors = set(np.argsort(scores,kind='stable')[-topk:].tolist())
            for values in hashes.values():
                dist = popcount[values ^ values[i]].sum(1).astype(np.int16)
                dist[i] = 100
                neighbors.update(np.argsort(dist,kind='stable')[:4].tolist())
            for n in neighbors:
                if i != n:
                    pairs.add(tuple(sorted((i,n))))
    return sorted(pairs), centers, bows


def geometry(a, b):
    """Mutual ratio SIFT matches + robust similarity transform + image correlation."""
    if min(len(a['desc']),len(b['desc'])) < 12:
        return dict(status='insufficient_features', matches=0)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = bf.knnMatch(a['desc'],b['desc'],k=2)
    backward = bf.knnMatch(b['desc'],a['desc'],k=2)
    reverse = {m.queryIdx:m.trainIdx for m,n in backward if m.distance < .75*n.distance}
    good = [m for m,n in forward if m.distance < .75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    result = dict(status='rejected', matches=len(good))
    if len(good) < 12:
        return result
    p = np.float32([a['xy'][m.queryIdx] for m in good])
    q = np.float32([b['xy'][m.trainIdx] for m in good])
    cv2.setRNGSeed(3407)
    affine, mask = cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3,
                                             maxIters=2000,confidence=.995,refineIters=10)
    if affine is None or mask is None:
        return result
    keep = mask.ravel().astype(bool)
    count = int(keep.sum())
    if count < 8:
        return result
    ratio = count/len(good)
    coverage = [float(cv2.contourArea(cv2.convexHull(z[keep]))/(512*512)) for z in (p,q)]
    scale = float(np.linalg.norm(affine[:,0]))
    errors = np.linalg.norm(p @ affine[:,:2].T+affine[:,2]-q,axis=1)
    small = affine.copy(); small[:,2] /= 2
    warped = cv2.warpAffine(a['gray'],small,(256,256))
    valid = cv2.warpAffine(np.ones((256,256),np.uint8),small,(256,256),flags=cv2.INTER_NEAREST)>0
    valid &= (warped>3)&(b['gray']>3)
    correlation = 0.
    if valid.sum()>=1024:
        x,y = warped[valid].astype(float),b['gray'][valid].astype(float)
        if min(x.std(),y.std()) > 3:
            correlation = float(np.corrcoef(x,y)[0,1])
    result.update(inliers=count, inlier_ratio=ratio, coverage=coverage, scale=scale,
                  median_error=float(np.median(errors[keep])), correlation=correlation,
                  affine=affine.tolist(), inlier_points_a=p[keep].tolist(), inlier_points_b=q[keep].tolist())
    plausible = .5<=scale<=2 and min(coverage)>=.04 and result['median_error']<=2
    if plausible and count>=20 and ratio>=.65 and min(coverage)>=.10 and correlation>=.75:
        result['status']='strong_overlap'
    elif plausible and count>=12 and ratio>=.45 and correlation>=.45:
        result['status']='review'
    return result


def grouped_split(groups, counts, seed, trials=2048, fraction=.1):
    names = sorted(counts)
    overall = sum((counts[n] for n in names),np.zeros(9,np.int64))
    presence = sum(((counts[n]>0).astype(int) for n in names),np.zeros(9,np.int64))
    target = round(len(names)*fraction)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        order = rng.permutation(len(groups))
        sizes = np.cumsum([len(groups[i]) for i in order])
        k = int(np.argmin(abs(sizes-target)))+1
        val = sorted(n for i in order[:k] for n in groups[i])
        if not 0<len(val)<len(names):
            continue
        pixels = sum((counts[n] for n in val),np.zeros(9,np.int64))
        pres = sum(((counts[n]>0).astype(int) for n in val),np.zeros(9,np.int64))
        if np.any((pixels[1:]==0)|(overall[1:]-pixels[1:]==0)):
            continue
        pixel_share = pixels[1:]/np.maximum(overall[1:],1)
        presence_share = pres[1:]/np.maximum(presence[1:],1)
        score = float(np.max(abs(pixel_share-fraction))+np.max(abs(presence_share-fraction))
                      +2*abs(len(val)/len(names)-fraction))
        if best is None or score<best[0]:
            best=(score,val,pixels,pres)
    if best is None:
        raise ValueError('No feasible grouped split; inspect oversized groups and rare classes')
    score,val,pixels,pres=best
    vset=set(val)
    return dict(status='candidate_not_scene_certified',seed=seed, selection_trials=trials,
                score=score,train=[n for n in names if n not in vset],val=val,
                note='Only observed strong-overlap components held together; unknown relations remain.'), dict(
                    overall_pixels=overall.tolist(), val_pixels=pixels.tolist(),
                    train_pixels=(overall-pixels).tolist(),val_presence=pres.tolist(),
                    train_presence=(presence-pres).tolist())


def panel(pair, features, names, path):
    i,j=pair['indices']
    canvas=Image.new('RGB',(1024,560),'white'); draw=ImageDraw.Draw(canvas)
    for side,idx in enumerate((i,j)):
        gray=Image.fromarray(features[idx]['gray']).resize((512,512)).convert('RGB')
        canvas.paste(gray,(512*side,25))
        draw.text((512*side+5,5),names[idx],fill='black')
    for p,q in zip(pair['inlier_points_a'][::max(1,pair['inliers']//20)],pair['inlier_points_b'][::max(1,pair['inliers']//20)]):
        draw.line((p[0],p[1]+25,q[0]+512,q[1]+25),fill='lime',width=1)
    draw.text((5,540),f"{pair['status']} inliers={pair['inliers']} corr={pair['correlation']:.3f}",fill='black')
    canvas.save(path)


def run(args):
    if cv2 is None:
        raise RuntimeError('Optional audit dependency missing: opencv-python-headless')
    cv2.setNumThreads(1)
    start=time.perf_counter(); output=Path(args.output)
    source=Path(args.features).read_bytes()
    rows=sorted([r for r in json.loads(source) if r['group'] in ('train','val')],key=lambda r:r['name'])
    names=[r['name'] for r in rows]
    if len(set(names))!=len(names) or not rows:
        raise ValueError('Invalid input manifest')
    config=dict(args=dict(vars(args)), manifest_sha256=hashlib.sha256(source).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),opencv=cv2.__version__)
    config['args'].pop('resume',None)
    if output.exists():
        if not args.resume or json.loads((output/'config.json').read_text(encoding='utf-8'))!=config:
            raise FileExistsError('Refusing existing output without matching explicit resume')
    else:
        output.mkdir(parents=True); write(output/'config.json',config)
    write(output/'run.json',dict(status='running',stage='extract',images=len(rows)))
    folder=output/'features'; folder.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(extract,[(r,Path(args.masks),folder) for r in rows]),total=len(rows),desc='SIFT + metadata + labels',mininterval=5))
    features=[]; meta=[]
    for n in names:
        with np.load(folder/(n+'.npz'),allow_pickle=False) as f:
            features.append({k:f[k].astype(np.float32) if k=='desc' else f[k] for k in ('xy','desc','gray','counts')})
            m=json.loads(str(f['metadata'])); meta.append(dict(name=n,**m))
    write(output/'metadata.json',meta)
    write(output/'run.json',dict(status='running',stage='retrieve'))
    if (output/'candidates.json').exists():
        pairs=json.loads((output/'candidates.json').read_text())
    else:
        pairs,centers,bows=retrieve(features,rows,args.seed,args.topk)
        np.savez_compressed(output/'retrieval.npz',centers=centers,bows=bows)
        write(output/'candidates.json',pairs)
    write(output/'run.json',dict(status='running',stage='geometry',candidate_pairs=len(pairs)))
    results=[]
    if (output/'geometry.jsonl').exists():
        results=[json.loads(line) for line in (output/'geometry.jsonl').read_text().splitlines() if line.strip()]
    done={tuple(r['indices']) for r in results}
    def check(pair):
        i,j=pair
        return dict(indices=[i,j],a=names[i],b=names[j],**geometry(features[i],features[j]))
    with (output/'geometry.jsonl').open('a',encoding='utf-8') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        remaining=[p for p in pairs if tuple(p) not in done]
        for r in tqdm(pool.map(check,remaining),total=len(remaining),desc='geometric verification',mininterval=5):
            results.append(r); log.write(json.dumps(r,allow_nan=False)+'\n'); log.flush()
    strong=[r for r in results if r['status']=='strong_overlap']
    review=[r for r in results if r['status']=='review']
    groups=components(names,[(r['a'],r['b']) for r in strong])
    write(output/'groups.json',[dict(group_id=f'g{i:04d}',members=g) for i,g in enumerate(groups)])
    write(output/'strong_pairs.json',strong); write(output/'review_pairs.json',review)
    split,balance=grouped_split(groups,{n:f['counts'] for n,f in zip(names,features)},args.seed)
    write(output/'candidate_split.json',split); write(output/'class_balance.json',balance)
    oldval={r['name'] for r in rows if r['group']=='val'}; newval=set(split['val'])
    crossed=[r for r in strong if (r['a'] in oldval)!=(r['b'] in oldval)]
    panels=output/'panels';panels.mkdir(exist_ok=True)
    chosen=sorted(crossed,key=lambda r:-r['inliers'])[:8]+sorted(strong,key=lambda r:r['correlation'])[:8]+review[:8]
    for r in chosen:
        panel(r,features,names,panels/(r['a']+'_'+r['b']+'.png'))
    summary=dict(images=len(rows),candidate_pairs=len(pairs),status_counts=dict(Counter(r['status'] for r in results)),
        groups=len(groups),non_singleton_groups=sum(len(g)>1 for g in groups),largest_groups=[len(g) for g in groups[:20]],
        old_cross_strong_pairs=len(crossed),old_val_images_with_strong_train_overlap=len({n for r in crossed for n in (r['a'],r['b']) if n in oldval}),
        candidate_train=len(split['train']),candidate_val=len(split['val']),
        candidate_cross_strong_pairs=sum((r['a'] in newval)!=(r['b'] in newval) for r in strong),
        candidate_cross_review_pairs=sum((r['a'] in newval)!=(r['b'] in newval) for r in review),
        metadata_nonempty_images=sum(bool(m['info'] or m['exif']) for m in meta),
        warnings=['Retrieval is approximate and limited to top-k; missing edges are not independent scenes.',
                  'Strong means heuristic geometric and photometric evidence, not human-confirmed source identity.',
                  'No official source IDs; proposed split is NOT approved for a scene-independent claim.',
                  'Test images excluded from graph, retrieval vocabulary, labels and split selection.',
                  'No original split, images, labels or weights changed; no training performed.'])
    write(output/'summary.json',summary)
    write(output/'run.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start,**summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('features','masks','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--topk',type=int,default=12)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
