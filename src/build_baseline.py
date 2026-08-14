#!/usr/bin/env python3
"""Prompting baseline: the direct channel and nothing else.

This is the reference point the rest of the system has to beat. It uses the
same backbone and the same few-shot direct prompt, reads the k=12 direct
samples that `run_inference.sh` drew anyway, and aggregates them by
self-consistency. No auxiliary elicitation channel, no verification probe, no
gate, and no fitted decision constant enters it.

Aggregation rules, all read off the task definition instead of fitted:
  numerics    cluster the parsed values at the metric's own 5% tolerance, take
              the largest cluster, answer with its median
  city        majority over parsed cities including the null answer; abstain
              when null wins
  set-valued  admit an object named in more than half of the samples that
              parsed

Writes to outputs/predictions_baseline.jsonl by default:

    python3 src/build_baseline.py
    python3 dataset2026/evaluate.py -p outputs/predictions_baseline.jsonl \
        -g dataset2026/data/test.jsonl
"""
import json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P

MD = P.OUT
DST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    P.OUT, "predictions_baseline.jsonl")
TEST = [json.loads(l) for l in open(os.path.join(P.DATA, "test.jsonl"))]
TOL = 0.05                      # the scorer's own tolerance, not a fitted value


def subs_of(rel):
    return [r["SubjectEntity"] for r in TEST if r["Relation"] == rel]


def cluster(vals, tol=TOL):
    vals = sorted(v for v in vals if v is not None and v > 0)
    cl = []
    for v in vals:
        for c in cl:
            if (abs(v - c[-1]) / max(v, c[-1]) <= tol
                    or abs(v - c[0]) / max(v, c[0]) <= tol):
                c.append(v)
                break
        else:
            cl.append([v])
    return cl


out = {}

# --------------------------- numerics: median of the largest cluster ---------
for rel in ("hasCapacity", "hasArea"):
    d = json.load(open(os.path.join(MD, f"raw_{rel}.json")))
    for s in subs_of(rel):
        vals = [v for t in d.get(s, [])
                if (v := P.num_from(t, rel, last=True)) is not None]
        cl = sorted(cluster(vals), key=len, reverse=True)
        if not cl:
            out[(s, rel)] = []
            continue
        c = sorted(cl[0])
        out[(s, rel)] = [P.fmt_num(c[len(c) // 2])]

# --------------------------- city: majority including the null answer --------
rel = "personHasCityOfDeath"
d = json.load(open(os.path.join(MD, f"raw_{rel}.json")))
for s in subs_of(rel):
    votes = []
    for t in d.get(s, []):
        c = P.city_from(t)
        if c != "__P__":                       # "__P__" marks an unparsed sample
            votes.append(P.norm(c) if c else "")
    if not votes:
        out[(s, rel)] = []
        continue
    top, _n = Counter(votes).most_common(1)[0]
    if not top:                                # the null answer won the vote
        out[(s, rel)] = []
        continue
    surf = Counter(c.strip() for t in d.get(s, [])
                   if (c := P.city_from(t)) not in (None, "", "__P__")
                   and P.norm(c) == top)
    out[(s, rel)] = [surf.most_common(1)[0][0]] if surf else []

# --------------------------- company: named in over half the samples ---------
rel = "companyTradesAtStockExchange"
d = json.load(open(os.path.join(MD, f"raw_{rel}.json")))
for s in subs_of(rel):
    cnt, sfm, n = Counter(), defaultdict(Counter), 0
    for t in d.get(s, []):
        o = P.objs_from(t, last=True)
        if o is None:
            continue
        n += 1
        for x in {P.norm(y) for y in o}:
            cnt[x] += 1
        for y in o:
            sfm[P.norm(y)][y] += 1
    out[(s, rel)] = [sfm[k].most_common(1)[0][0]
                     for k, c in cnt.items() if n and c > n / 2]

# borders: the list pass is the direct prompt for this relation, and its
# checkpoint already stores a per-object sample frequency
rel = "countryLandBordersCountry"
freq = json.load(open(os.path.join(MD, "raw_borders_freq.json")))
for s in subs_of(rel):
    out[(s, rel)] = [surf for _k, (f, surf) in freq.get(s, {}).items() if f > 0.5]

# awards: the enumeration checkpoint stores per-name counts over the passes,
# so majority is a count above half the largest count seen for that subject
rel = "awardWonBy"
aw = json.load(open(os.path.join(MD, "raw_award.json")))
for s in subs_of(rel):
    src = aw.get(s)
    if not src:
        out[(s, rel)] = []
        continue
    counts = src["counts"]
    hi = max(counts.values()) if counts else 0
    out[(s, rel)] = [max(src["sf"][k], key=src["sf"][k].get)
                     for k, c in counts.items() if hi and c > hi / 2]

with open(DST, "w") as f:
    for r in TEST:
        f.write(json.dumps({
            "SubjectEntity": r["SubjectEntity"],
            "Relation": r["Relation"],
            "ObjectEntities": out.get((r["SubjectEntity"], r["Relation"]), []),
        }, ensure_ascii=False) + "\n")
print(f"wrote {DST}: {len(TEST)} rows")
