#!/usr/bin/env python3
"""Single-backbone system on Muse-Glimmer-30B (29.6B, inside the 32B budget
on its own - no room for a second model, so no small-model agreement term).

Runs the full elicitation + probe stack on the TEST subjects and dumps raw
intermediates for the offline decision layer, mirroring the instrumented
Mistral runs so the two are directly comparable.

The one adaptation this model forces: it answers in two channels, reasoning
("to=self") then the user-facing answer. A generation prompt therefore starts
in the reasoning channel, which destroys one-token Yes/No probing - the next
token is the start of deliberation, not "Yes"/"No". Appending the answer-
channel header ourselves forces the model to answer immediately, which is
what the probes need. Sampling channels keep the reasoning, since that is
where this model's factual advantage comes from.
"""
import argparse, json, math, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P

MUSE = "meta-models/Muse-Glimmer-30B"
SPLIT = os.environ.get("MUSE_SPLIT", "test")
OUT = os.path.join(P.ROOT, "outputs" if SPLIT == "test"
                   else f"outputs_{SPLIT}")
os.makedirs(OUT, exist_ok=True)
TEST = [json.loads(l) for l in open(os.path.join(P.DATA, f"{SPLIT}.jsonl"))]
ANSWER_CHANNEL = " to=user<|message|>"


def subjects(rel):
    return [r["SubjectEntity"] for r in TEST if r["Relation"] == rel]


class Muse(P.HF):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MUSE, use_fast=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # Single 80GB card: the whole 59.5GB model fits with ~20GB left for
        # activations - no sharding, no CPU offload, no meta-device traps.
        # On two 32GB cards, split the layer stack explicitly and pin the
        # norm/rotary/lm_head with the upper half; auto device maps put
        # lm_head on CPU, which turns every decode step into a 202k-vocab
        # CPU matmul.
        n = torch.cuda.device_count()
        big = torch.cuda.get_device_properties(0).total_memory > 70e9
        if n == 1 or big:
            dm = {"": 0}
        else:
            dm = {"model.vision_tower": 0, "model.vision_adapter": 0,
                  "model.vision_projection": 0, "model.perception_emb_norm": 0,
                  "model.language_model.embed_tokens": 0}
            for i in range(52):
                dm[f"model.language_model.layers.{i}"] = 0 if i < 24 else 1
            dm["model.language_model.norm"] = 1
            dm["model.language_model.rotary_emb"] = 1
            dm["lm_head"] = 1
        kw = dict(dtype=torch.bfloat16, device_map=dm)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(MUSE, **kw)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(MUSE, **kw)
        self.model.eval()
        self._letters = {}
        for w in ("Yes", "No", "A", "B"):
            ids = set()
            for v in (w, " " + w, w.lower(), " " + w.lower()):
                t = self.tok.encode(v, add_special_tokens=False)
                if t:
                    ids.add(t[0])
            self._letters[w] = list(ids)

    def _tmpl(self, p, strength, answer_now):
        s = self.tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, reasoning_strength=strength)
        if self.tok.bos_token and s.startswith(self.tok.bos_token):
            s = s[len(self.tok.bos_token):]
        return s + ANSWER_CHANNEL if answer_now else s

    def chat(self, prompts, strength="low", answer_now=False):
        return [self._tmpl(p, strength, answer_now) for p in prompts]

    def probe_yes(self, prompts, batch=8):
        """One-token Yes/No, forced into the answer channel."""
        return self.p_yes(self.chat(prompts, "low", answer_now=True), batch)


def ckpt(name, obj=None):
    p = f"{OUT}/{name}.json"
    if obj is None:
        return json.load(open(p)) if os.path.exists(p) else None
    json.dump(obj, open(p, "w"), ensure_ascii=False)
    print(f"  [ckpt] {name}", flush=True)


def stage_sample(hf):
    """Direct + recitation channels for the four sampled relations."""
    plan = [("hasCapacity", 6, 1024), ("hasArea", 6, 1024),
            ("personHasCityOfDeath", 6, 1024),
            ("companyTradesAtStockExchange", 6, 1024)]
    for rel, k, mt in plan:
        if ckpt(f"raw_{rel}") is not None:
            print(f"[{rel}] cached", flush=True)
            continue
        ss = subjects(rel)
        print(f"[sample:{rel}] n={len(ss)} k={k}", flush=True)
        outs = hf.sample(hf.chat([P.build_prompt(rel, s) for s in ss]),
                         k, mt, 1.0, batch=1, stop_q=False)
        ckpt(f"raw_{rel}", dict(zip(ss, outs)))


def stage_probe(hf):
    """Border and exchange verification, forced into the answer channel."""
    if ckpt("raw_borders") is None:
        subs = subjects("countryLandBordersCountry")
        print("[borders] universe", flush=True)
        uni_raw = hf.sample(hf.chat([P.COUNTRY_UNI], "low", answer_now=True),
                            3, 2000, 0.8, batch=1, stop_q=False)[0]
        names, surf = Counter(), {}
        for txt in uni_raw:
            for line in txt.splitlines():
                nm = line.strip().strip("-*0123456789. ").strip()
                if nm and len(nm) <= 45 and ":" not in nm:
                    k = P.norm(nm)
                    surf.setdefault(k, nm)
                    names[k] += 1
        universe = [surf[k] for k, c in names.items() if c >= 2]
        print(f"[borders] universe={len(universe)}", flush=True)
        d = P.DEFS["countryLandBordersCountry"]
        pairs, prompts = [], []
        for s in subs:
            for c in universe:
                if P.norm(c) == P.norm(s):
                    continue
                pairs.append((s, c, "fw"))
                prompts.append(P.BORDER_PROBE.format(d=d, a=s, b=c))
                pairs.append((s, c, "bw"))
                prompts.append(P.BORDER_PROBE.format(d=d, a=c, b=s))
        print(f"[borders] {len(prompts)} probes", flush=True)
        ps = hf.probe_yes(prompts, batch=8)
        ckpt("raw_borders", {"universe": universe,
                             "pairs": [list(p) for p in pairs], "ps": ps})
    if ckpt("raw_gates") is None:
        cs = subjects("personHasCityOfDeath")
        gd = hf.probe_yes([P.GATE_CITY.format(s=s) for s in cs])
        co = subjects("companyTradesAtStockExchange")
        gc = hf.probe_yes([P.GATE_COMP.format(s=s) for s in co])
        ckpt("raw_gates", {"city": dict(zip(cs, gd)),
                           "company": dict(zip(co, gc))})


def stage_award(hf):
    if ckpt("raw_award") is not None:
        return
    res = {}
    for s in subjects("awardWonBy"):
        print(f"[award] {s}", flush=True)
        counts, sf = Counter(), defaultdict(Counter)

        def absorb(lists, boost=1):
            for objs in lists:
                if not objs:
                    continue
                for o in set(objs):
                    counts[P.norm(o)] += boost
                    sf[P.norm(o)][o] += 1

        # 3000-token generations OOM alongside 29.6B of weights; the answer
        # channel makes long enumerations cheap without the deliberation.
        absorb([P.objs_from(t, last=True) for t in hf.sample(
            hf.chat([P.build_prompt("awardWonBy", s)], "low", answer_now=True),
            2, 600, 1.0, batch=1, stop_q=False)[0]])
        absorb([P.objs_from(t, last=True) for t in hf.sample(
            hf.chat([P.AWARD_BOOST.format(s=s)], "low", answer_now=True),
            2, 600, 1.1, batch=1, stop_q=False)[0]])
        yrs = list(range(1950, 2027))
        outs = hf.sample(hf.chat([P.AWARD_YR.format(s=s, y=y) for y in yrs],
                                 "low", answer_now=True),
                         1, 128, 0.8, batch=1, stop_q=False)
        for ts in outs:
            for o in (P.objs_from(ts[0], last=True) or []):
                counts[P.norm(o)] += 2
                sf[P.norm(o)][o] += 1
        res[s] = {"counts": dict(counts),
                  "sf": {k: dict(v) for k, v in sf.items()}}
    ckpt("raw_award", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["sample", "probe", "award", "canary"])
    a = ap.parse_args()
    hf = Muse()
    if a.stage == "canary":
        # does forcing the answer channel actually yield a Yes/No token?
        pr = [P.GATE_CITY.format(s="Albert Einstein"),
              P.GATE_CITY.format(s="Taylor Swift")]
        print("forced-channel p_yes(deceased):",
              [round(x, 3) for x in hf.probe_yes(pr)], flush=True)
        print("unforced p_yes(deceased):",
              [round(x, 3) for x in hf.p_yes(hf.chat(pr))], flush=True)
        txt = hf.sample(hf.chat([P.GATE_CITY.format(s="Albert Einstein")],
                                "low", answer_now=True), 1, 24, 0.0,
                        batch=1, stop_q=False)[0][0]
        print("forced-channel raw:", repr(txt[:120]), flush=True)
    else:
        {"sample": stage_sample, "probe": stage_probe,
         "award": stage_award}[a.stage](hf)
    print("stage complete", flush=True)
