"""
build_splits.py

Combines annotations.jsonl (5k) and jiotv_augmented_3000_annotations.jsonl (3k)
into a single deduplicated pool, then creates an 80 / 10 / 10 train / val / test
split, stratified so that:
  - the typo-heavy "augmented" examples are spread across all three splits
    (not dumped entirely into one), since the model needs to learn typo
    tolerance during training AND be evaluated on it
  - rows with entity_type='language' or 'director' (rare, ~3% and ~1.5% of
    the combined data, present only in the 5k file) are spread across all
    three splits too, so val/test aren't missing label classes
  - the 12 exact-duplicate queries that appear in both files are deduped
    (kept once, from the 5k/original file)

WHY THIS SPLIT:
  - 8000 total examples is small for fine-tuning a generative seq2seq /
    instruction model; 80/10/10 (6400 / 800 / 800) keeps enough train data
    while still giving statistically meaningful val/test sets (800 examples
    each easily catches per-entity-type accuracy regressions, since the
    rarest class - director - has ~120 examples total, ~12-15 per split).
  - Stratifying on (has_language, has_director, is_augmented_typo) ensures
    val/test are representative of BOTH clean queries (5k) and typo/noisy
    queries (3k), which is what the model will see in production.

Output files (JSONL, same {"search_query":..., "Output":[...]} schema):
  train.jsonl  (~6400 rows)
  val.jsonl    (~800 rows)
  test.jsonl   (~800 rows)
"""

import json
import random
from collections import defaultdict

random.seed(42)

A_PATH = '/home/claude/annotations.jsonl'                       # 5k
B_PATH = '/home/claude/jiotv_augmented_3000_annotations.jsonl'  # 3k

a = [json.loads(l) for l in open(A_PATH, encoding='utf-8')]
b = [json.loads(l) for l in open(B_PATH, encoding='utf-8')]

for x in a:
    x['_source'] = '5k'
for x in b:
    x['_source'] = '3k_augmented'

seen = set()
pool = []
for x in a + b:
    q = x['search_query']
    if q in seen:
        continue
    seen.add(q)
    pool.append(x)

print(f"Combined pool (deduped): {len(pool)} rows")


def strat_key(x):
    has_lang = any(e['entity_type'] == 'language' for e in x['Output'])
    has_dir = any(e['entity_type'] == 'director' for e in x['Output'])
    return (x['_source'], has_lang, has_dir)


buckets = defaultdict(list)
for x in pool:
    buckets[strat_key(x)].append(x)

train, val, test = [], [], []
for key, rows in buckets.items():
    random.shuffle(rows)
    n = len(rows)
    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    val.extend(rows[:n_val])
    test.extend(rows[n_val:n_val + n_test])
    train.extend(rows[n_val + n_test:])

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")


def dump(rows, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            out = {"search_query": r["search_query"], "Output": r["Output"]}
            f.write(json.dumps(out, ensure_ascii=False) + '\n')


dump(train, '/home/claude/train.jsonl')
dump(val, '/home/claude/val.jsonl')
dump(test, '/home/claude/test.jsonl')

# sanity: entity-type coverage per split
from collections import Counter


def coverage(rows, name):
    c = Counter()
    for r in rows:
        for e in r['Output']:
            c[e['entity_type']] += 1
    print(name, dict(c))


coverage(train, 'train')
coverage(val, 'val')
coverage(test, 'test')
