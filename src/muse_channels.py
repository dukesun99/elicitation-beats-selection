#!/usr/bin/env python3
"""Completes the single-Muse port: the pipeline components missing from
muse_system.py (extra elicitation channels, exchange verification, capacity
duels and PMI).

Design point. Forcing the answer channel (the trick the probe canary
validated) also makes auxiliary channels cheap: they answer immediately
instead of deliberating, so a channel costs ~100 tokens rather than ~1000.
The primary direct channel keeps its reasoning, since that is where Muse's
factual advantage lives; the auxiliary channels exist for diversity and
verification, where speed matters more than deliberation.
"""
import argparse, json, math, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P
from muse_elicit import scale_k, K_SCALE, Muse, OUT, TEST, subjects, ckpt


# ------------------- auxiliary elicitation channels -------------------
def stage_channels(hf):
    """recite / cited / wiki / disambig / greedy, all forced-answer."""
    jobs = {
        "hasCapacity": [
            ("recite", lambda s: P.PRE + "\n\n" + P.RECITE_NUM["hasCapacity"].format(s=s), 4, 200),
            ("cited", lambda s: P.PRE + "\n\n" + P.CAP_CITE.format(s=s), 4, 120),
            ("wiki", lambda s: P.WIKI_NUM["hasCapacity"].format(
                s=__import__("re").split(r"\s+in\s+", s)[0],
                hint=(__import__("re").split(r"\s+in\s+", s)[-1] if " in " in s else "")), 4, 48),
            ("greedy", lambda s: P.build_prompt("hasCapacity", s), 1, 120),
        ],
        "hasArea": [
            ("recite", lambda s: P.PRE + "\n\n" + P.RECITE_NUM["hasArea"].format(s=s), 4, 200),
            ("wiki", lambda s: P.WIKI_NUM["hasArea"].format(
                s=__import__("re").sub(r",.*$", "", s)), 4, 48),
            ("disambig", lambda s: P.PRE + "\n\n" + P.AREADIS.format(
                s=s, hint=P.area_hint(s)), 4, 120),
            ("greedy", lambda s: P.build_prompt("hasArea", s), 1, 120),
        ],
        "personHasCityOfDeath": [
            ("recite", lambda s: P.PRE + "\n\n" + P.CITY_RECITE.format(s=s), 4, 220),
            ("ml", lambda s: P.PRE + "\n\n" + P.CITY_ML.format(s=s), 4, 220),
            ("presup", lambda s: P.PRE + "\n\n" + P.CITY_PRESUP.format(s=s), 4, 200),
        ],
    }
    for rel, chans in jobs.items():
        name = f"raw_chan_{rel}"
        if ckpt(name) is not None:
            print(f"[chan:{rel}] cached", flush=True)
            continue
        ss = subjects(rel)
        pools = defaultdict(list)
        for label, mk, k, mt in chans:
            k = scale_k(k)
            print(f"[chan:{rel}:{label}] n={len(ss)} k={k}", flush=True)
            temp = 1.0 if k > 1 else 0.0
            outs = hf.sample(hf.chat([mk(s) for s in ss], "low", answer_now=True),
                             k, mt, temp, batch=8, stop_q=False)
            for s, ts in zip(ss, outs):
                for t in ts:
                    pools[s].append([label, t])
        ckpt(name, {s: pools[s] for s in ss})


# ------------------- exchange verification -------------------
def stage_exchange(hf):
    if ckpt("raw_exchange") is not None:
        print("[exchange] cached", flush=True)
        return
    subs = subjects("companyTradesAtStockExchange")
    print("[exchange] universe", flush=True)
    uni_raw = hf.sample(hf.chat([P.EXCH_UNI], "low", answer_now=True),
                        3, 1200, 0.8, batch=6, stop_q=False)[0]
    names, surf = Counter(), {}
    for txt in uni_raw:
        for line in txt.splitlines():
            nm = line.strip().strip("-*0123456789. ").strip()
            if nm and len(nm) <= 60 and ":" not in nm:
                k = P.norm(nm)
                surf.setdefault(k, nm)
                names[k] += 1
    universe = [surf[k] for k, c in names.items() if c >= 2]
    TRAIN = [json.loads(l) for l in open(os.path.join(P.DATA, "train.jsonl"))]
    placebos = [r["SubjectEntity"] for r in TRAIN
                if r["Relation"] == "companyTradesAtStockExchange"
                and not r["ObjectEntities"]][:3]
    print(f"[exchange] universe={len(universe)}; placebo", flush=True)
    pl = hf.probe_yes([P.EXCH_PROBE.format(s=pb, e=e)
                       for e in universe for pb in placebos])
    plc = {P.norm(e): sum(pl[i * 3:(i + 1) * 3]) / 3
           for i, e in enumerate(universe)}
    print(f"[exchange] {len(subs) * len(universe)} probes", flush=True)
    probes = hf.probe_yes([P.EXCH_PROBE.format(s=s, e=e)
                           for s in subs for e in universe], batch=8)
    ckpt("raw_exchange", {"universe": universe, "placebo": plc,
                          "probes": probes})


# ------------------- capacity duels + PMI -------------------
def stage_numeric_terms(hf):
    if ckpt("raw_capterms") is not None:
        print("[capterms] cached", flush=True)
        return
    raw = ckpt("raw_hasCapacity")
    if raw is None:
        print("[capterms] needs raw_hasCapacity; skipping", flush=True)
        return
    subs = subjects("hasCapacity")
    reps = {}
    for s in subs:
        vals = [v for t in raw.get(s, [])
                if (v := P.num_from(t, "hasCapacity", last=True)) is not None]
        cl = sorted(P.cluster(vals), key=len, reverse=True)[:4]
        rr = []
        for c in cl:
            rp = max(c)
            if all(abs(rp - x) / max(rp, x) > 0.05 for x in rr):
                rr.append(rp)
        reps[s] = rr
    dp, dk = [], []
    for s, rr in reps.items():
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                for (x, y) in ((rr[i], rr[j]), (rr[j], rr[i])):
                    dk.append((s, x, y))
                    dp.append(P.DUEL.format(s=s, a=f"{int(x):,}", b=f"{int(y):,}"))
    print(f"[capterms] {len(dp)} duels", flush=True)
    da = hf.p_a(hf.chat(dp, "low", answer_now=True), batch=2) if dp else []
    duels = defaultdict(list)
    for (s, x, y), p in zip(dk, da):
        duels[s].append([x, y, p])
    pk, pc, pu = [], [], []
    for s, rr in reps.items():
        for rp in rr:
            vs = str(int(rp))
            pk.append((s, rp))
            pc.append((hf.chat([P.build_prompt("hasCapacity", s)], "low",
                                answer_now=True)[0] + ' {"value": ', vs))
            pu.append(('Q: What is the maximum spectator capacity of a large '
                       'sports venue? A: {"value": ', vs))
    print(f"[capterms] {len(pk)} PMI pairs", flush=True)
    # long few-shot prefixes: batch 1 is what fits in the headroom
    lc = hf.seq_logprob(pc, batch=8) if pk else []
    lu = hf.seq_logprob(pu, batch=8) if pk else []
    pmi = defaultdict(list)
    for (s, rp), a, b in zip(pk, lc, lu):
        pmi[s].append([rp, a - b])
    ckpt("raw_capterms", {"duels": {s: duels.get(s, []) for s in subs},
                          "pmi": {s: pmi.get(s, []) for s in subs}})


# ------------------- borders frequency channel -------------------
def stage_borders_freq(hf):
    """The port gave borders probes but no sampling channel, so it scored on
    half the signal Mistral fuses. Forced-answer keeps it cheap."""
    if ckpt("raw_borders_freq") is not None:
        print("[borders_freq] cached", flush=True)
        return
    rel = "countryLandBordersCountry"
    ss = subjects(rel)
    print(f"[borders_freq] n={len(ss)}", flush=True)
    outs = []
    for seed in ((13, 29) if K_SCALE < 1.5 else (13, 29, 51, 83)):
        outs.append(hf.sample(
            hf.chat([P.build_prompt(rel, s, seed=seed) for s in ss],
                    "low", answer_now=True),
            scale_k(3), 320, 1.0, batch=8, stop_q=False))
    freq = {}
    for i, s in enumerate(ss):
        cnt, sf, nn = Counter(), defaultdict(Counter), 0
        for grp in outs:
            for t in grp[i]:
                o = P.objs_from(t, last=True)
                if o is None:
                    continue
                nn += 1
                for x in {P.norm(y) for y in o}:
                    cnt[x] += 1
                for y in o:
                    sf[P.norm(y)][y] += 1
        freq[s] = {k: [v / max(nn, 1), sf[k].most_common(1)[0][0]]
                   for k, v in cnt.items()}
    ckpt("raw_borders_freq", freq)


# ------------------- boosted passes for Muse's two weak relations -------------------
def stage_boost(hf):
    """Muse lost borders and awards on sampling volume, not capability: its
    borders frequency channel ran k=3 x 2 seeds against Mistral's larger
    merged pool, and its award stage was cut to 2x600 sequences to survive
    OOM. Both get a second, larger pass here, checkpointed separately so the
    pools can be merged offline."""
    rel = "countryLandBordersCountry"
    if ckpt("raw_borders_freq2") is None:
        ss = subjects(rel)
        print(f"[boost:borders] n={len(ss)}", flush=True)
        outs = []
        for seed in (7, 41, 97):
            outs.append(hf.sample(
                hf.chat([P.build_prompt(rel, s, seed=seed) for s in ss],
                        "low", answer_now=True),
                4, 320, 1.0, batch=8, stop_q=False))
        freq = {}
        for i, s in enumerate(ss):
            cnt, sf, nn = Counter(), defaultdict(Counter), 0
            for grp in outs:
                for t in grp[i]:
                    o = P.objs_from(t, last=True)
                    if o is None:
                        continue
                    nn += 1
                    for x in {P.norm(y) for y in o}:
                        cnt[x] += 1
                    for y in o:
                        sf[P.norm(y)][y] += 1
            freq[s] = {k: [v / max(nn, 1), sf[k].most_common(1)[0][0]]
                       for k, v in cnt.items()}
        ckpt("raw_borders_freq2", freq)
    if ckpt("raw_award2") is None:
        res = {}
        for s in subjects("awardWonBy"):
            print(f"[boost:award] {s}", flush=True)
            counts, sf = Counter(), defaultdict(Counter)
            def absorb(lists, boost=1):
                for objs in lists:
                    if not objs:
                        continue
                    for o in set(objs):
                        counts[P.norm(o)] += boost
                        sf[P.norm(o)][o] += 1
            # short prompts + few parallel sequences: the combination that
            # finally fit alongside 28.2GB of weights on one card
            for _rep in range(3):
                absorb([P.objs_from(t, last=True) for t in hf.sample(
                    hf.chat([P.AWARD_BOOST.format(s=s)], "low", answer_now=True),
                    2, 600, 1.1, batch=6, stop_q=False)[0]])
            for rnd in range(2):
                found = [sf[k].most_common(1)[0][0]
                         for k, _ in counts.most_common(120)]
                absorb([P.objs_from(t, last=True) for t in hf.sample(
                    hf.chat([P.AWARD_EXCL.format(
                        s=s, found=json.dumps(found, ensure_ascii=False))],
                        "low", answer_now=True),
                    2, 600, 1.2, batch=6, stop_q=False)[0]])
            res[s] = {"counts": dict(counts),
                      "sf": {k: dict(v) for k, v in sf.items()}}
        ckpt("raw_award2", res)


# ------------------- high-volume forced-answer sampling -------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["channels", "exchange", "capterms",
                             "borders_freq", "boost"])
    a = ap.parse_args()
    hf = Muse()
    {"channels": stage_channels, "exchange": stage_exchange,
     "capterms": stage_numeric_terms,
     "borders_freq": stage_borders_freq,
     "boost": stage_boost}[a.stage](hf)
    print("stage complete", flush=True)
