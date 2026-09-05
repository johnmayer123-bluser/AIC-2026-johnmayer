"""Read-only image duplication and appearance audit. Never reads test labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import platform
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
COSINE = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32)[None, :] + 1) / 64)
BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def index_images(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            if path.stem in result:
                raise ValueError(f'Duplicate stem: {path.stem}')
            result[path.stem] = path
    if not result:
        raise ValueError(f'No images: {root}')
    return result


def image_hashes(image):
    gray = image.convert('L')
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coeff = (COSINE @ small @ COSINE.T).reshape(-1)
    bits = coeff > np.median(coeff[1:])
    bits[0] = False  # Exclude average brightness (DC).
    d = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return np.packbits(bits).tobytes().hex(), np.packbits(d[:, 1:] > d[:, :-1]).tobytes().hex()


def extract(task):
    group, name, path = task
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as loaded:
        image = loaded.convert('RGB')
    pixels = image.tobytes()
    exact = hashlib.sha256(str(image.size).encode() + b'\0' + pixels).hexdigest()
    phash, dhash = image_hashes(image)
    rgb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64) / 255
    brightness = rgb.mean(axis=2)
    nonblack = rgb.max(axis=2) > 2/255
    gray = rgb @ np.array([.299, .587, .114])
    lap = (-4*gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    features = dict(
        r_mean=float(rgb[..., 0].mean()), g_mean=float(rgb[..., 1].mean()),
        b_mean=float(rgb[..., 2].mean()), brightness=float(brightness.mean()),
        contrast=float(brightness.std()), color_spread=float((rgb.max(2)-rgb.min(2)).mean()),
        black_fraction=float((~nonblack).mean()),
        nonblack_brightness=float(brightness[nonblack].mean()) if nonblack.any() else None,
        laplacian_variance=float(lap.var()),
    )
    return dict(group=group, name=name, path=str(path.resolve()), size=list(image.size),
                file_sha256=hashlib.sha256(raw).hexdigest(), rgb_sha256=exact,
                phash=phash, dhash=dhash, features=features)


def exact_cross(rows):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['rgb_sha256']].append(dict(group=row['group'], name=row['name']))
    return [items for items in by_hash.values() if len({i['group'] for i in items}) > 1]


def hash_distances(query, references):
    return BIT_COUNTS[np.bitwise_xor(query, references)].sum(1)


def nearest(query_rows, train_rows, threshold=8):
    train_bits = {key: np.stack([np.frombuffer(bytes.fromhex(r[key]), np.uint8) for r in train_rows])
                  for key in ('phash', 'dhash')}
    neighbors, candidates = [], []
    for row in query_rows:
        distances = {key: hash_distances(np.frombuffer(bytes.fromhex(row[key]), np.uint8), values)
                     for key, values in train_bits.items()}
        nearest_ids = {key: int(np.argmin(values)) for key, values in distances.items()}
        neighbors.append(dict(name=row['name'], group=row['group'], **{
            key: dict(train_name=train_rows[index]['name'], distance=int(distances[key][index]))
            for key, index in nearest_ids.items()}))
        indices = set()
        for values in distances.values():
            indices.update(np.argsort(values, kind='stable')[:3].tolist())
        for index in sorted(indices):
            pd, dd = int(distances['phash'][index]), int(distances['dhash'][index])
            if min(pd, dd) > threshold:
                continue
            ref = train_rows[index]
            candidates.append(dict(query_group=row['group'], query_name=row['name'],
                                   train_name=ref['name'], phash_distance=pd, dhash_distance=dd,
                                   exact_rgb=row['rgb_sha256']==ref['rgb_sha256'],
                                   low_texture=min(row['features']['contrast'], ref['features']['contrast']) < .02))
    return neighbors, sorted(candidates, key=lambda r: (r['phash_distance']+r['dhash_distance'], r['query_name']))


def summarize_features(rows):
    result = {}
    for key in rows[0]['features']:
        values = np.array([r['features'][key] for r in rows if r['features'][key] is not None])
        result[key] = dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                           std=float(values.std()) if len(values) else None,
                           quantiles_05_25_50_75_95=np.quantile(values, [.05,.25,.5,.75,.95]).tolist() if len(values) else None)
    return result


def pair_panel(pair, query_index, train_index, path):
    canvas = Image.new('RGB', (960, 535), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, row) in enumerate([(pair['query_group'], query_index[pair['query_name']]),
                                       ('train', train_index[pair['train_name']])]):
        with Image.open(row['path']) as loaded:
            image = loaded.convert('RGB')
        draw.text((i*480+5, 5), f"{label}: {row['name']}", fill='black')
        image.thumbnail((480,480))
        canvas.paste(image, (i*480,25))
    draw.text((5,510), f"pHash distance={pair['phash_distance']}, dHash distance={pair['dhash_distance']}; candidate only, not proof of duplication", fill='black')
    canvas.save(path)


def run(args):
    start = time.perf_counter()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if args.workers < 1 or not 0 <= args.threshold <= 64:
        raise ValueError('Invalid workers or hash threshold')
    split_bytes = Path(args.split).read_bytes()
    split = json.loads(split_bytes)
    images, test = index_images(args.images), index_images(args.test_images)
    for key in ('train', 'val'):
        if not split[key] or len(split[key]) != len(set(split[key])):
            raise ValueError(f'Empty or duplicated split.{key}')
    if set(split['train']) & set(split['val']) or set(images) != set(split['train']) | set(split['val']):
        raise ValueError('Split overlap or image coverage mismatch')
    tasks = [(group, name, images[name]) for group in ('train','val') for name in sorted(split[group])]
    tasks += [('test', name, path) for name, path in sorted(test.items())]
    output.mkdir(parents=True)
    def write(name, payload):
        (output/name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    metadata = dict(status='running', args=vars(args), split_sha256=hashlib.sha256(split_bytes).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__)
    write('run.json', metadata)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(tqdm(executor.map(extract, tasks), total=len(tasks), desc='image audit', mininterval=5))
    write('image_features.json', rows)
    grouped = {key: [r for r in rows if r['group']==key] for key in ('train','val','test')}
    appearance = {key: summarize_features(values) for key, values in grouped.items()}
    effects = {group: {key: (appearance[group][key]['mean']-base['mean'])/base['std']
                        if base['std'] and appearance[group][key]['mean'] is not None else None
                        for key, base in appearance['train'].items()} for group in ('val','test')}
    exact = exact_cross(rows)
    write('exact_cross_split.json', exact)
    comparisons = {}
    for group in ('val','test'):
        neighbors, candidates = nearest(grouped[group], grouped['train'], args.threshold)
        write(f'{group}_nearest_train.json', neighbors)
        write(f'{group}_candidate_pairs.json', candidates)
        comparisons[group] = dict(query_images=len(neighbors), candidate_pairs=len(candidates),
            queries_with_candidate=len({p['query_name'] for p in candidates}),
            distance_summary={key: dict(mean=float(np.mean([r[key]['distance'] for r in neighbors])),
                median=float(np.median([r[key]['distance'] for r in neighbors])),
                queries_at_most_4=sum(r[key]['distance']<=4 for r in neighbors),
                queries_at_most_8=sum(r[key]['distance']<=8 for r in neighbors)) for key in ('phash','dhash')})
        folder = output/f'{group}_pairs'
        folder.mkdir()
        qindex = {r['name']:r for r in grouped[group]}
        tindex = {r['name']:r for r in grouped['train']}
        for index, pair in enumerate(candidates[:12]):
            pair_panel(pair, qindex, tindex, folder/f"{index:02d}_{pair['query_name']}_{pair['train_name']}.png")
    metadata.update(status='complete', elapsed_seconds=time.perf_counter()-start)
    report = dict(counts={k:len(v) for k,v in grouped.items()}, appearance=appearance,
                  standardized_mean_difference_vs_train=effects, exact_cross_split_groups=len(exact),
                  nearest_train=comparisons, notes=[
                      'Exact RGB SHA256 includes dimensions; file hashes are also saved.',
                      'pHash/dHash screen whole-image similarity only; threshold8 is heuristic, not a duplicate verdict.',
                      'Only top3 matches per hash per query retained as candidates; not an exhaustive graph.',
                      'Hashes can miss adjacent/translated/rotated patches; no candidates does not prove scene independence.',
                      'Flat textures and black padding can cause hash collisions; manually inspect panels.',
                      'Appearance features at128x128; brightness=RGB mean, black=maxRGB<=2/255.',
                      'Laplacian variance reflects texture and scale as well as sharpness, not blur alone.',
                      'Distribution differences are descriptive, not proof of the cause of the score gap.',
                      'No labels, training, image deletion, or split changes in this audit.'])
    write('summary.json', report)
    write('run.json', metadata)
    print(json.dumps(dict(status='complete', counts=report['counts'], exact_groups=len(exact),
                          nearest_train=comparisons, output=str(output.resolve())), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('images','test-images','split','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threshold', type=int, default=8)
    run(parser.parse_args())
