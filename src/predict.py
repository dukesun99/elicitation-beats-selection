#!/usr/bin/env python3
"""Closed-book knowledge base construction for LM-KBC 2026.

Closed-book: all facts come from the model's parameters. All frozen decision
constants are inlined below. Three stages:

  python3 predict.py --stage mistral    # main backbone: channels + probes (GPU)
  python3 predict.py --stage qwen4b     # small-model agreement pass for hasArea
  python3 predict.py --stage finalize   # fuse + decision rules -> predictions

Outputs land in <repo>/outputs/: per-relation checkpoints (resumable) and the
final predictions.jsonl in test-set order. Sampling channels use temperature
sampling and are stochastic run-to-run; scores reproduce within about one
macro-F1 point. Everything downstream of the checkpoints is deterministic.

Relations and their frozen pipelines:
  countryLandBordersCountry  L3 k8 + model-generated country universe +
                             bidirectional Yes/No probes; sigma = .8 freq +
                             .2 pair; tau .55; integral-territory mapping
  personHasCityOfDeath       direct k12 + recitation k6 + multilingual k8 +
                             presupposition k8; alive-gate .15; empty_tau .3;
                             share tau .5; force-answer at gate >= .7
  companyTradesAtStockExchange  direct k10 + exchange-universe probes with
                             placebo yes-bias correction; sigma = .7 freq +
                             .3 ver; tau .5
  hasCapacity                direct k12 + recitation k6 + greedy + cited k6 +
                             wiki-table k6 (weight 4) + similarity k6; cluster
                             vote (5% linkage, highest rep) + pairwise-duel
                             term (.5) + PMI term (.5); overshoot scale .95
  hasArea                    direct k12 + recitation k6 + greedy + wiki-table
                             k6 (weight 2) + type-disambiguation k6 +
                             similarity k6; median rep; + .3 agreement bonus
                             when Qwen3-4B's top cluster concurs
  awardWonBy                 direct enum k6 + per-year 1950-2026 (count boost
                             2, first-year filter) + exclusion x2 + complete-
                             ness k10 + verified-CoT k4; union count >= 2;
                             never empty
"""
import argparse, json, math, os, random, re, unicodedata
from collections import Counter, defaultdict

ROOT = os.environ.get("AKBC_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "dataset2026", "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

MISTRAL = "unsloth/Mistral-Small-3.2-24B-Instruct-2506"
QWEN4B = "Qwen/Qwen3-4B-Instruct-2507"

# ----------------------------- text utils -----------------------------
APO = set("'’‘ʻʼʹ`´")
SYM = set("+$<=>|~^")


def norm(s):
    s = "".join(c for c in s.strip() if c not in APO)
    s = unicodedata.normalize("NFKD", s).casefold()
    out = []
    for c in s:
        if c in APO or unicodedata.combining(c):
            continue
        out.append(" " if c in SYM or unicodedata.category(c).startswith("P") else c)
    return " ".join("".join(out).split())


def parse_number(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", v.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


UNIT = {"km2": 1.0, "km^2": 1.0, "km²": 1.0, "sq km": 1.0, "sqkm": 1.0,
        "square kilometres": 1.0, "square kilometers": 1.0,
        "sq mi": 2.58999, "square miles": 2.58999, "mi2": 2.58999,
        "hectare": 0.01, "hectares": 0.01, "ha": 0.01,
        "m2": 1e-6, "m^2": 1e-6, "acre": 0.00404686, "acres": 0.00404686}


def to_km2(v, unit):
    if v is None:
        return None
    u = (unit or "km2").strip().lower()
    f = UNIT.get(u)
    if f is None:
        f = next((x for k, x in UNIT.items() if k in u), 1.0)
    return v * f


def cluster(vals, tol=0.05):
    vals = sorted(v for v in vals if v is not None and v > 0)
    cl = []
    for v in vals:
        for c in cl:
            if abs(v - c[-1]) / max(v, c[-1]) <= tol or abs(v - c[0]) / max(v, c[0]) <= tol:
                c.append(v)
                break
        else:
            cl.append([v])
    return cl


def fmt_num(x):
    if x >= 1000:
        return str(int(round(x)))
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s or "0"


def extract_json(text, last=False):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    spans, i = [], 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[i:j + 1])
                    break
        i += 1
    for s in (reversed(spans) if last else spans):
        try:
            return json.loads(s)
        except Exception:
            continue
    return None


# ----------------------------- task text -----------------------------
DEFS = {
    "countryLandBordersCountry": ("Countries that share a LAND border with the "
        "subject country. Only land borders count; maritime boundaries do NOT "
        "count. Island states with no land border have an empty answer. Only "
        "currently recognised states. Borders through a country's integral "
        "overseas territory count (e.g. France-Suriname via French Guiana)."),
    "personHasCityOfDeath": ("The CITY where the person died, at city granularity "
        "(not the country, state, or region). If the person is still alive as of "
        "1 July 2026, or the city of death is not publicly known, the answer is empty."),
    "awardWonBy": ("All entities (people, groups, organizations) that have RECEIVED "
        "this exact award in history. Recipients, not winning works. Predecessor or "
        "successor awards with similar names are DIFFERENT awards."),
    "companyTradesAtStockExchange": ("The stock exchange(s) where the company's "
        "shares are publicly traded now. Multiple listings possible. If not listed "
        "in its own name (private, delisted, unlisted subsidiary), answer empty."),
    "hasCapacity": ("The maximum spectator capacity of the venue as an integer "
        "number of people. If several figures were published (seated vs total, "
        "before/after renovation), the HIGHEST published capacity counts."),
    "hasArea": ("The surface area of the geographic entity in square kilometres "
        "(km2). For countries, TOTAL area including inland water. Convert sq mi "
        "(x2.58999) or hectares (x0.01) if recalled in those units."),
}
PRE = ("You are completing a knowledge base. Answer from your own knowledge of the "
       "world as of 1 July 2026. Follow the relation definition exactly and answer "
       "with JSON only, no other text.")
SCHEMA = {
    "countryLandBordersCountry": '{"objects": ["Country1", ...]} ({"objects": []} if none)',
    "personHasCityOfDeath": '{"city": "CityName"} or {"city": null}',
    "awardWonBy": '{"objects": ["Name1", ...]}',
    "companyTradesAtStockExchange": '{"objects": ["Exchange full official name", ...]} ([] if not listed)',
    "hasCapacity": '{"value": <integer>, "unit": "people"}',
    "hasArea": '{"value": <number>, "unit": "km2"}',
}
QUESTION = {
    "countryLandBordersCountry": "List all countries that share a land border with {s}.",
    "personHasCityOfDeath": "In which city did {s} die?",
    "awardWonBy": 'List ALL recipients of the award "{s}" in history.',
    "companyTradesAtStockExchange": ('On which stock exchange(s) is {s} currently '
        'listed? Use full official exchange names (e.g., "New York Stock Exchange").'),
    "hasCapacity": "What is the maximum spectator capacity of {s}? Use the highest published figure.",
    "hasArea": "What is the total area of {s} in square kilometres (including inland water, for countries)?",
}

TRAIN = [json.loads(l) for l in open(os.path.join(DATA, "train.jsonl"))]
TEST = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))]


def subjects(rel):
    return [r["SubjectEntity"] for r in TEST if r["Relation"] == rel]


def gold_canon(row):
    return [(e[0] if isinstance(e, list) else e) for e in row["ObjectEntities"]]


def exemplar_answer(rel, row):
    objs = gold_canon(row)
    if rel == "personHasCityOfDeath":
        return json.dumps({"city": objs[0] if objs else None}, ensure_ascii=False)
    if rel in ("hasCapacity", "hasArea"):
        v = float(objs[0]) if objs else 0
        unit = "people" if rel == "hasCapacity" else "km2"
        return json.dumps({"value": int(v) if v == int(v) else v, "unit": unit},
                          ensure_ascii=False)
    return json.dumps({"objects": objs}, ensure_ascii=False)


def fewshot(rel, n=8, seed=13, similar_to=None):
    rows = [r for r in TRAIN if r["Relation"] == rel]
    empt = [r for r in rows if not r["ObjectEntities"]]
    full = [r for r in rows if r["ObjectEntities"]]
    if rel == "awardWonBy":
        full = sorted(full, key=lambda r: len(r["ObjectEntities"]))[:12]
    if similar_to:
        def sim(a, b):
            ta, tb = set(a.lower().split()), set(b.lower().split())
            sc = len(ta & tb)
            for grp in (("lake", "lough", "loch"), ("island", "isla", "isle"),
                        ("stadium", "arena", "field", "park", "oval")):
                if any(w in ta for w in grp) and any(w in tb for w in grp):
                    sc += 2
            return sc
        full = sorted(full, key=lambda r: -sim(similar_to, r["SubjectEntity"]))
        picked = full[:n - 2] + empt[:2]
    else:
        rng = random.Random(seed)
        picked = (rng.sample(empt, min(2, len(empt))) if empt else [])
        picked += rng.sample(full, min(n - len(picked), len(full)))
        rng.shuffle(picked)
    return picked


def build_prompt(rel, subject, seed=13, similar=False):
    parts = [PRE, "", f"Relation definition: {DEFS[rel]}", "",
             f"Output JSON schema: {SCHEMA[rel]}", "", "Examples:"]
    for ex in fewshot(rel, seed=seed, similar_to=(subject if similar else None)):
        parts.append("Q: " + QUESTION[rel].format(s=ex["SubjectEntity"]))
        parts.append("A: " + exemplar_answer(rel, ex))
    parts += ["", "Q: " + QUESTION[rel].format(s=subject), "A:"]
    return "\n".join(parts)


# ----------------------------- HF backend -----------------------------
class HF:
    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="auto")
        except ValueError:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self._letters = {}
        for w in ("Yes", "No", "A", "B"):
            ids = set()
            for v in (w, " " + w, w.lower(), " " + w.lower()):
                t = self.tok.encode(v, add_special_tokens=False)
                if t:
                    ids.add(t[0])
            self._letters[w] = list(ids)

    def sample(self, prompts, n, max_new, temp=1.0, batch=2, stop_q=True):
        torch = self.torch
        outs = [[] for _ in prompts]
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(self.model.device)
            remaining = n
            while remaining > 0:
                k = min(remaining, 4)
                remaining -= k
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc, do_sample=temp > 0,
                        temperature=temp if temp > 0 else None,
                        top_p=0.95 if temp > 0 else None,
                        num_return_sequences=k, max_new_tokens=max_new,
                        pad_token_id=self.tok.pad_token_id)
                texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
                for j in range(len(chunk)):
                    for t in texts[j * k:(j + 1) * k]:
                        outs[i + j].append(t.split("\nQ:")[0] if stop_q else t)
                del gen
                torch.cuda.empty_cache()
            if (i // batch) % 10 == 0:
                print(f"    {min(i + batch, len(prompts))}/{len(prompts)}", flush=True)
        return outs

    def _two_way(self, prompts, pos, neg, batch=8):
        torch = self.torch
        res = []
        for i in range(0, len(prompts), batch):
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            for row in logits:
                a = max(float(row[t]) for t in self._letters[pos])
                b = max(float(row[t]) for t in self._letters[neg])
                res.append(math.exp(a) / (math.exp(a) + math.exp(b)))
        return res

    def p_yes(self, prompts, batch=8):
        return self._two_way(prompts, "Yes", "No", batch)

    def p_a(self, prompts, batch=8):
        return self._two_way(prompts, "A", "B", batch)

    def seq_logprob(self, pairs, batch=4):
        """Summed logprob of continuation given prefix (teacher-forced)."""
        torch = self.torch
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            rows, clens = [], []
            for pre, cont in chunk:
                pi = self.tok.encode(pre)
                ci = self.tok.encode(cont, add_special_tokens=False)
                rows.append(pi + ci)
                clens.append(len(ci))
            maxlen = max(len(x) for x in rows)
            pad = self.tok.pad_token_id
            inp = torch.tensor([[pad] * (maxlen - len(x)) + x for x in rows],
                               device=self.model.device)
            att = torch.tensor([[0] * (maxlen - len(x)) + [1] * len(x)
                                for x in rows], device=self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids=inp, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            for j, (ids, cl) in enumerate(zip(rows, clens)):
                nfull = len(ids)
                s = 0.0
                for k in range(nfull - cl, nfull):
                    pos = maxlen - nfull + k
                    s += float(lp[j, pos - 1, ids[k]])
                out.append(s)
        return out


def ckpt(name, obj=None):
    p = os.path.join(OUT, name + ".json")
    if obj is None:
        return json.load(open(p)) if os.path.exists(p) else None
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"  [ckpt] {name}", flush=True)


# ----------------------------- channels: prompts -----------------------------
CITY_RECITE = ("From memory, write the 2-3 sentence passage describing the death of "
               "{s}. If this person is alive as of 1 July 2026 or you do not know "
               "where they died, say so. Then on the FINAL line output JSON: "
               "{{\"city\": \"CityName\"}} or {{\"city\": null}}.")
CITY_ML = ("Consider {s}. In which language is this person's biography most likely "
           "best documented? Recall what LOCAL sources in that language say about "
           "where this person died. Then output the ENGLISH city name as JSON on "
           "the final line: {{\"city\": \"CityName\"}} or {{\"city\": null}}.")
CITY_PRESUP = ("It is documented that {s} has died. Recall the SPECIFIC city, town, "
               "or municipality where {s} died - the exact locality, not the metro "
               "area or region. Output JSON on the final line: "
               "{{\"city\": \"Name\"}} or {{\"city\": null}} ONLY if truly unrecorded.")
GATE_CITY = ("As of 1 July 2026, is {s} deceased? Answer with exactly one word: "
             "Yes or No.\nAnswer:")
GATE_COMP = ("As of 1 July 2026, is {s} a publicly traded company whose own shares "
             "are listed on a stock exchange (not private, not delisted, and not "
             "merely an unlisted subsidiary)? Answer with exactly one word: Yes or "
             "No.\nAnswer:")
BORDER_PROBE = ("Relation definition: {d}\n\nDoes {a} share a land border with {b}? "
                "Only land borders count; maritime boundaries do not. Answer with "
                "exactly one word: Yes or No.\nAnswer:")
COUNTRY_UNI = ("List every currently recognised sovereign state in the world, one "
               "name per line, using common English short names. No numbering, no "
               "commentary, just one country name per line.")
EXCH_UNI = ("List the stock exchanges of the world where publicly traded companies "
            "are commonly listed, one FULL OFFICIAL name per line (e.g. \"New York "
            "Stock Exchange\", not \"NYSE\"). No numbering, no commentary.")
EXCH_PROBE = ("As of 1 July 2026, are shares of {s} publicly traded on the {e}? "
              "Answer with exactly one word: Yes or No.\nAnswer:")
RECITE_NUM = {
    "hasCapacity": ("Recall what you know about {s}: when it was built, renovations, "
                    "and published capacity figures. Think briefly, then on the FINAL "
                    "line output the HIGHEST published maximum spectator capacity as "
                    "JSON: {{\"value\": <integer>, \"unit\": \"people\"}}"),
    "hasArea": ("Recall what you know about {s}, including its total area. Convert "
                "sq mi (x2.58999) or hectares (x0.01) to km2 if needed. Think "
                "briefly, then on the FINAL line output JSON: "
                "{{\"value\": <number>, \"unit\": \"km2\"}}"),
}
CAP_CITE = ("What maximum spectator capacity is {s} most commonly cited with in "
            "reference works? Use the highest published figure. Output JSON only: "
            "{{\"value\": <integer>, \"unit\": \"people\"}}")
WIKI_NUM = {
    "hasCapacity": "{s}\nType: sports venue\n| Field | Value |\n|---|---|\n| Location | {hint} |\n| Capacity | ",
    "hasArea": "{s}\nType: geographic entity\n| Field | Value |\n|---|---|\n| Area | ",
}
AREADIS = ("What is the total area of {s} in square kilometres? Important: give "
           "the area of {hint}. Output JSON only: "
           "{{\"value\": <number>, \"unit\": \"km2\"}}")
DUEL = ("Which is closer to the true maximum spectator capacity of {s}? "
        "(A) {a} people, or (B) {b} people. "
        "Answer with exactly one letter: A or B.\nAnswer: (")
AWARD_YR = ("Who received the award \"{s}\" in {y}? This is the exact award, not a "
            "predecessor or successor award. Output JSON only: "
            "{{\"objects\": [\"Name1\", ...]}} (empty list if not awarded in {y}).")
AWARD_EXCL = ("Recipients of the award \"{s}\" found so far: {found}.\nList OTHER "
              "recipients of this exact award NOT in the list above. Output JSON "
              "only: {{\"objects\": [\"Name1\", ...]}} (empty list if none).")
AWARD_BOOST = ("List as many recipients of the award \"{s}\" as you can recall - "
               "aim for COMPLETENESS across all decades, including less famous "
               "recipients. Output JSON only: {{\"objects\": [\"Name1\", ...]}}")
AWARD_META = "In which year was the award \"{s}\" first awarded? Answer with only the year."
AWARD_COT_FS = (
    'Q: List ALL recipients of the award "Fields medal" in history.\n'
    "<think>The Fields Medal is awarded every four years in mathematics. Rather "
    "than recalling all winners at once, I enumerate by ceremony year. 1990: "
    "Vladimir Drinfeld, Vaughan Jones, Shigefumi Mori, Edward Witten. 1994: Jean "
    "Bourgain, Pierre-Louis Lions, Jean-Christophe Yoccoz, Efim Zelmanov. 1998: "
    "Richard Borcherds, Timothy Gowers, Maxim Kontsevich, Curtis T. McMullen."
    "</think>\n"
    '{"objects": ["Vladimir Drinfeld", "Vaughan Jones", "Shigefumi Mori", '
    '"Edward Witten", "Jean Bourgain", "Pierre-Louis Lions", '
    '"Jean-Christophe Yoccoz", "Efim Zelmanov", "Richard Borcherds", '
    '"Timothy Gowers", "Maxim Kontsevich", "Curtis T. McMullen"]}')
TERRITORY_MAP = {"french guiana": "France", "ceuta": "Spain", "melilla": "Spain"}


def area_hint(s):
    sl = s.lower()
    if any(w in sl for w in ("lake", "lough", "loch")):
        return "the lake itself (water surface), not its basin or municipality"
    if any(w in sl for w in ("island", "isla", "isle", "ile")) or "," in s:
        return "the island itself, not the municipality, province, or archipelago"
    return "the exact entity named"


def city_from(t):
    d = extract_json(t, last=True) or extract_json(t)
    if isinstance(d, dict):
        c = d.get("city")
        return c if isinstance(c, str) and c.strip() else None
    return "__P__"


def objs_from(t, last=False):
    d = extract_json(t, last=last) or extract_json(t)
    if isinstance(d, dict) and isinstance(d.get("objects"), list):
        return [x.strip() for x in d["objects"] if isinstance(x, str) and x.strip()]
    return None


def num_from(t, rel, last=True, wiki=False):
    if wiki:
        t0 = t.split("|")[0].split("\n")[0]
        v = parse_number(t0)
        if v is None:
            return None
        tl = t0.lower()
        if "sq mi" in tl or "mile" in tl:
            v *= 2.58999
        elif "hectare" in tl:
            v *= 0.01
        return v
    d = extract_json(t, last=last) or extract_json(t)
    if isinstance(d, dict):
        v = parse_number(d.get("value"))
        return to_km2(v, d.get("unit")) if rel == "hasArea" else v
    return None


# ----------------------------- stage: mistral -----------------------------
def stage_mistral():
    hf = HF(MISTRAL)

    # ===== countryLandBordersCountry =====
    if ckpt("borders") is None:
        subs = subjects("countryLandBordersCountry")
        print("[borders] universe", flush=True)
        uni_raw = hf.sample([COUNTRY_UNI], 5, 2500, 0.8, batch=1, stop_q=False)[0]
        names, surf = Counter(), {}
        for txt in uni_raw:
            for line in txt.splitlines():
                nm = line.strip().strip("-*0123456789. ").strip()
                if nm and len(nm) <= 45 and ":" not in nm:
                    k = norm(nm)
                    surf.setdefault(k, nm)
                    names[k] += 1
        universe = [surf[k] for k, c in names.items() if c >= 2]
        print(f"[borders] universe={len(universe)}; sampling", flush=True)
        freq = {}
        outs = hf.sample([build_prompt("countryLandBordersCountry", s, seed=sd)
                          for s in subs for sd in (13, 29)], 4, 512, 1.0, batch=2)
        for i, s in enumerate(subs):
            cnt, sf, nn = Counter(), defaultdict(Counter), 0
            for ts in (outs[2 * i], outs[2 * i + 1]):
                for t in ts:
                    o = objs_from(t)
                    if o is None:
                        continue
                    nn += 1
                    for x in {norm(y) for y in o}:
                        cnt[x] += 1
                    for y in o:
                        sf[norm(y)][y] += 1
            freq[s] = {k: (v / max(nn, 1),
                           sf[k].most_common(1)[0][0]) for k, v in cnt.items()}
        print("[borders] probes", flush=True)
        d = DEFS["countryLandBordersCountry"]
        pairs, prompts = [], []
        for s in subs:
            for c in universe:
                if norm(c) == norm(s):
                    continue
                pairs.append((s, c, "fw"))
                prompts.append(BORDER_PROBE.format(d=d, a=s, b=c))
                pairs.append((s, c, "bw"))
                prompts.append(BORDER_PROBE.format(d=d, a=c, b=s))
        print(f"[borders] {len(prompts)} probes", flush=True)
        ps = hf.p_yes(prompts, batch=8)
        pair_score = defaultdict(dict)
        acc = defaultdict(lambda: [0.0, 0])
        for (s, c, _dr), p in zip(pairs, ps):
            a = acc[(s, norm(c))]
            a[0] += p
            a[1] += 1
        for (s, ck), (tot, nn) in acc.items():
            pair_score[s][ck] = tot / nn
        results = {}
        for s in subs:
            cand = dict(freq.get(s, {}))
            for ck in pair_score.get(s, {}):
                cand.setdefault(ck, (0.0, next((u for u in universe
                                                if norm(u) == ck), ck)))
            out = []
            for ck, (fr, surface) in cand.items():
                pr = pair_score.get(s, {}).get(ck)
                sigma = 0.8 * fr + 0.2 * pr if pr is not None else fr
                if sigma >= 0.55:
                    out.append((sigma, surface))
            out.sort(key=lambda x: -x[0])
            mapped, seen = [], set()
            for _sc, o in out:
                o2 = TERRITORY_MAP.get(norm(o), o)
                if norm(o2) not in seen:
                    seen.add(norm(o2))
                    mapped.append(o2)
            results[s] = mapped
            print(f"  {s} -> {mapped}", flush=True)
        ckpt("borders", results)

    # ===== personHasCityOfDeath =====
    if ckpt("city") is None:
        subs = subjects("personHasCityOfDeath")
        print("[city] gates", flush=True)
        gates = dict(zip(subs, hf.p_yes([GATE_CITY.format(s=s) for s in subs])))
        pools = defaultdict(list)
        for tmpl, k, mt, direct in ((None, 12, 96, True),
                                    (CITY_RECITE, 6, 320, False),
                                    (CITY_ML, 8, 400, False),
                                    (CITY_PRESUP, 8, 200, False)):
            if direct:
                prompts = [build_prompt("personHasCityOfDeath", s) for s in subs]
            else:
                prompts = [PRE + "\n\nRelation definition: " +
                           DEFS["personHasCityOfDeath"] + "\n\n" + tmpl.format(s=s)
                           for s in subs]
            print(f"[city] channel k={k}", flush=True)
            outs = hf.sample(prompts, k, mt, 1.0, batch=3, stop_q=direct)
            for s, ts in zip(subs, outs):
                for t in ts:
                    c = city_from(t)
                    if c != "__P__":
                        pools[s].append(c)
        results = {}
        for s in subs:
            p_dead = gates[s]
            samples = pools[s]
            ans = []
            if p_dead >= 0.15 and samples:
                n = len(samples)
                empty_frac = sum(1 for c in samples if c is None) / n
                cnt = Counter(norm(c) for c in samples if c)
                sf = defaultdict(Counter)
                for c in samples:
                    if c:
                        sf[norm(c)][c.strip()] += 1
                if cnt:
                    top, topc = cnt.most_common(1)[0]
                    if (empty_frac < 0.3 and topc / n >= 0.5) or p_dead >= 0.7:
                        ans = [sf[top].most_common(1)[0][0]]
            results[s] = ans
            print(f"  {s} p_dead={p_dead:.2f} -> {ans}", flush=True)
        ckpt("city", results)

    # ===== companyTradesAtStockExchange =====
    if ckpt("company") is None:
        subs = subjects("companyTradesAtStockExchange")
        print("[company] universe", flush=True)
        uni_raw = hf.sample([EXCH_UNI], 5, 1200, 0.8, batch=1, stop_q=False)[0]
        names, surf = Counter(), {}
        for txt in uni_raw:
            for line in txt.splitlines():
                nm = line.strip().strip("-*0123456789. ").strip()
                if nm and len(nm) <= 60 and ":" not in nm:
                    k = norm(nm)
                    surf.setdefault(k, nm)
                    names[k] += 1
        universe = [surf[k] for k, c in names.items() if c >= 2]
        placebos = [r["SubjectEntity"] for r in TRAIN
                    if r["Relation"] == "companyTradesAtStockExchange"
                    and not r["ObjectEntities"]][:3]
        print(f"[company] universe={len(universe)}; placebo probes", flush=True)
        pl = hf.p_yes([EXCH_PROBE.format(s=pb, e=e)
                       for e in universe for pb in placebos])
        plc = {norm(e): sum(pl[i * 3:(i + 1) * 3]) / 3
               for i, e in enumerate(universe)}
        print("[company] sampling", flush=True)
        outs = hf.sample([build_prompt("companyTradesAtStockExchange", s)
                          for s in subs], 10, 160, 1.0, batch=3)
        print("[company] probes", flush=True)
        probes = hf.p_yes([EXCH_PROBE.format(s=s, e=e)
                           for s in subs for e in universe], batch=8)
        results = {}
        for i, s in enumerate(subs):
            cnt, sf, nn = Counter(), defaultdict(Counter), 0
            for t in outs[i]:
                o = objs_from(t)
                if o is None:
                    continue
                nn += 1
                for x in {norm(y) for y in o}:
                    cnt[x] += 1
                for y in o:
                    sf[norm(y)][y] += 1
            ver = {}
            for j, e in enumerate(universe):
                k = norm(e)
                p = probes[i * len(universe) + j]
                pb = plc.get(k, 0.5)
                eps = 1e-6
                lg = (math.log(max(p, eps) / max(1 - p, eps)) -
                      math.log(max(pb, eps) / max(1 - pb, eps)))
                ver[k] = 1 / (1 + math.exp(-lg))
            ans = []
            for k in set(cnt) | set(ver):
                fr = cnt.get(k, 0) / max(nn, 1)
                sigma = 0.7 * fr + 0.3 * ver[k] if k in ver else fr
                if sigma >= 0.5:
                    ssurf = (sf[k].most_common(1)[0][0] if sf.get(k)
                             else next((u for u in universe if norm(u) == k), k))
                    ans.append((sigma, ssurf))
            ans.sort(key=lambda x: -x[0])
            results[s] = [a for _, a in ans]
            print(f"  {s} -> {results[s]}", flush=True)
        ckpt("company", results)

    # ===== hasCapacity =====
    if ckpt("capacity") is None:
        subs = subjects("hasCapacity")
        pools = defaultdict(list)
        for label, prompts, k, w, temp, stopq, wiki, last in (
            ("direct", [build_prompt("hasCapacity", s) for s in subs], 12, 1, 1.0, True, False, False),
            ("similar", [build_prompt("hasCapacity", s, similar=True) for s in subs], 6, 1, 1.0, True, False, False),
            ("recite", [PRE + "\n\n" + RECITE_NUM["hasCapacity"].format(s=s) for s in subs], 6, 1, 1.0, False, False, True),
            ("cited", [PRE + "\n\n" + CAP_CITE.format(s=s) for s in subs], 6, 1, 1.0, False, False, True),
            ("greedy", [build_prompt("hasCapacity", s) for s in subs], 1, 1, 0.0, True, False, False),
            ("wiki", [WIKI_NUM["hasCapacity"].format(
                s=re.split(r"\s+in\s+", s)[0],
                hint=(re.split(r"\s+in\s+", s)[-1] if " in " in s else ""))
                for s in subs], 6, 4, 1.0, False, True, False)):
            print(f"[capacity:{label}]", flush=True)
            mt = 24 if wiki else (400 if label in ("recite",) else 96)
            outs = hf.sample(prompts, k, mt, temp, batch=3, stop_q=stopq)
            for s, ts in zip(subs, outs):
                for t in ts:
                    v = num_from(t, "hasCapacity", last=last, wiki=wiki)
                    if v is not None:
                        pools[s].extend([v] * w)
        # duel probes on top-4 cluster reps
        print("[capacity] duels", flush=True)
        duel_pairs, duel_prompts = [], []
        for s in subs:
            cl = sorted(cluster(pools[s]), key=len, reverse=True)[:4]
            reps = []
            for c in cl:
                rp = max(c)
                if all(abs(rp - x) / max(rp, x) > 0.05 for x in reps):
                    reps.append(rp)
            for ai in range(len(reps)):
                for bi in range(ai + 1, len(reps)):
                    for (x, y) in ((reps[ai], reps[bi]), (reps[bi], reps[ai])):
                        duel_pairs.append((s, x, y))
                        duel_prompts.append(DUEL.format(
                            s=s, a=f"{int(x):,}", b=f"{int(y):,}"))
        duel_p = hf.p_a(duel_prompts) if duel_prompts else []
        duels = defaultdict(list)
        for (s, x, y), p in zip(duel_pairs, duel_p):
            duels[s].append((x, y, p))
        # PMI scoring of reps
        print("[capacity] PMI", flush=True)
        pmi_pairs_c, pmi_pairs_u, pmi_keys = [], [], []
        for s in subs:
            cl = sorted(cluster(pools[s]), key=len, reverse=True)[:4]
            for c in cl:
                rp = max(c)
                vs = str(int(rp))
                pmi_keys.append((s, rp))
                pmi_pairs_c.append((build_prompt("hasCapacity", s) + ' {"value": ', vs))
                pmi_pairs_u.append((
                    "Q: What is the maximum spectator capacity of a large sports "
                    'venue? A: {"value": ', vs))
        lc = hf.seq_logprob(pmi_pairs_c) if pmi_keys else []
        lu = hf.seq_logprob(pmi_pairs_u) if pmi_keys else []
        pmi = defaultdict(list)
        for (s, rp), a, b in zip(pmi_keys, lc, lu):
            pmi[s].append((rp, a - b))
        results = {}
        for s in subs:
            pool = pools[s]
            if not pool:
                results[s] = ["20000"]     # never abstain on numerics
                continue
            cl = sorted(cluster(pool), key=len, reverse=True)
            total = len(pool)
            plist = pmi.get(s, [])
            zsum = sum(math.exp(p) for _v, p in plist) or 1.0
            def duel_score(rep):
                ws, nn = 0.0, 0
                for x, y, p in duels.get(s, []):
                    if abs(x - rep) / max(x, rep) <= 0.05:
                        ws += p
                        nn += 1
                    elif abs(y - rep) / max(y, rep) <= 0.05:
                        ws += 1 - p
                        nn += 1
                return ws / nn if nn else 0.5
            best, bs = None, -1e9
            for c in cl:
                rep = max(c)
                pm = max((math.exp(p) / zsum for v, p in plist
                          if abs(v - rep) / max(v, rep) <= 0.05), default=0.0)
                sc = len(c) / total + 0.5 * duel_score(rep) + 0.5 * pm
                if sc > bs or (sc == bs and best is not None and rep > best):
                    bs, best = sc, rep
            results[s] = [fmt_num(best * 0.95)]
            print(f"  {s} -> {results[s]}", flush=True)
        ckpt("capacity", results)

    # ===== hasArea (pools only; agreement bonus applied in finalize) =====
    if ckpt("area_pools") is None:
        subs = subjects("hasArea")
        pools = defaultdict(list)
        for label, prompts, k, w, temp, stopq, wiki, last in (
            ("direct", [build_prompt("hasArea", s) for s in subs], 12, 1, 1.0, True, False, False),
            ("similar", [build_prompt("hasArea", s, similar=True) for s in subs], 6, 1, 1.0, True, False, False),
            ("recite", [PRE + "\n\n" + RECITE_NUM["hasArea"].format(s=s) for s in subs], 6, 1, 1.0, False, False, True),
            ("greedy", [build_prompt("hasArea", s) for s in subs], 1, 1, 0.0, True, False, False),
            ("wiki", [WIKI_NUM["hasArea"].format(s=re.sub(r",.*$", "", s)) for s in subs], 6, 2, 1.0, False, True, False),
            ("disambig", [PRE + "\n\n" + AREADIS.format(s=s, hint=area_hint(s)) for s in subs], 6, 1, 1.0, False, False, True)):
            print(f"[area:{label}]", flush=True)
            mt = 24 if wiki else (400 if label == "recite" else 96)
            outs = hf.sample(prompts, k, mt, temp, batch=3, stop_q=stopq)
            for s, ts in zip(subs, outs):
                for t in ts:
                    v = num_from(t, "hasArea", last=last, wiki=wiki)
                    if v is not None:
                        pools[s].extend([v] * w)
        ckpt("area_pools", {s: pools[s] for s in subs})

    # ===== awardWonBy =====
    if ckpt("award") is None:
        subs = subjects("awardWonBy")
        results = {}
        for s in subs:
            counts, sf = Counter(), defaultdict(Counter)
            def absorb(lists, boost=1):
                for objs in lists:
                    if objs is None:
                        continue
                    for o in set(objs):
                        counts[norm(o)] += boost
                        sf[norm(o)][o] += 1
            print(f"[award] {s}", flush=True)
            absorb([objs_from(t) for t in hf.sample(
                [build_prompt("awardWonBy", s)], 6, 3000, 1.0, batch=1)[0]])
            absorb([objs_from(t) for t in hf.sample(
                [AWARD_BOOST.format(s=s)], 10, 2048, 1.1, batch=1)[0]])
            cotp = (PRE + "\n\nReason inside <think></think>, then output ONLY the "
                    "JSON.\n\n" + AWARD_COT_FS + "\n\nQ: " +
                    QUESTION["awardWonBy"].format(s=s) + "\n")
            absorb([objs_from(t.split("</think>")[-1], last=True) for t in
                    hf.sample([cotp], 4, 1536, 1.0, batch=1, stop_q=False)[0]])
            meta_ts = hf.sample([AWARD_META.format(s=s)], 3, 12, 0.5,
                                batch=1, stop_q=False)[0]
            ys = [int(m.group(1)) for t in meta_ts
                  if (m := re.search(r"\b(1[89]\d\d|20[0-2]\d)\b", t))]
            fy = sorted(ys)[len(ys) // 2] if ys else None
            outs = hf.sample([AWARD_YR.format(s=s, y=y)
                              for y in range(1950, 2027)], 2, 128, 0.8,
                             batch=4, stop_q=False)
            for y, ts in zip(range(1950, 2027), outs):
                if fy and y < fy:
                    continue
                names = set()
                for t in ts:
                    o = objs_from(t, last=True)
                    if o:
                        names.update(o)
                for o in names:
                    counts[norm(o)] += 2
                    sf[norm(o)][o] += 1
            for _rnd in range(2):
                found = [sf[k].most_common(1)[0][0]
                         for k, _ in counts.most_common(150)]
                absorb([objs_from(t, last=True) for t in hf.sample(
                    [AWARD_EXCL.format(s=s, found=json.dumps(found, ensure_ascii=False))],
                    4, 2048, 1.2, batch=1, stop_q=False)[0]])
            kept = [(c, sf[k].most_common(1)[0][0])
                    for k, c in counts.items() if c >= 2]
            if not kept and counts:
                k0 = counts.most_common(1)[0][0]
                kept = [(counts[k0], sf[k0].most_common(1)[0][0])]
            kept.sort(key=lambda x: -x[0])
            results[s] = [x for _, x in kept]
            print(f"  {s} -> {len(results[s])} recipients (fy={fy})", flush=True)
        ckpt("award", results)
    print("stage mistral complete")


# ----------------------------- stage: qwen4b -----------------------------
def stage_qwen4b():
    hf = HF(QWEN4B)
    subs = subjects("hasArea")
    outs = hf.sample([build_prompt("hasArea", s) for s in subs], 6, 96, 1.0, batch=4)
    pools = {}
    for s, ts in zip(subs, outs):
        vals = [v for t in ts
                if (v := num_from(t, "hasArea", last=False)) is not None]
        pools[s] = vals
    ckpt("area_q4b", pools)
    print("stage qwen4b complete")


# ----------------------------- stage: finalize -----------------------------
def stage_finalize():
    borders = ckpt("borders")
    city = ckpt("city")
    company = ckpt("company")
    capacity = ckpt("capacity")
    area_pools = ckpt("area_pools")
    area_q4b = ckpt("area_q4b") or {}
    award = ckpt("award")
    area = {}
    for s, vals in area_pools.items():
        cl = sorted(cluster(vals), key=len, reverse=True)
        if not cl:
            area[s] = ["100"]
            continue
        total = len(vals)
        xcl = sorted(cluster(area_q4b.get(s, [])), key=len, reverse=True)
        xrep = sorted(xcl[0])[len(xcl[0]) // 2] if xcl else None
        best, bs = None, -1
        for c in cl:
            rep = sorted(c)[len(c) // 2]
            agree = 1.0 if (xrep and rep > 0 and
                            abs(xrep - rep) / max(xrep, rep) <= 0.05) else 0.0
            sc = len(c) / total + 0.3 * agree
            if sc > bs:
                bs, best = sc, rep
        area[s] = [fmt_num(best)]
    table = {"countryLandBordersCountry": borders, "personHasCityOfDeath": city,
             "companyTradesAtStockExchange": company, "hasCapacity": capacity,
             "hasArea": area, "awardWonBy": award}
    out, miss = [], []
    for r in TEST:
        v = table.get(r["Relation"], {}).get(r["SubjectEntity"])
        if v is None:
            miss.append((r["SubjectEntity"], r["Relation"]))
            v = []
        out.append({"SubjectEntity": r["SubjectEntity"], "Relation": r["Relation"],
                    "ObjectEntities": v})
    assert not miss, f"missing predictions: {miss[:5]}"
    with open(os.path.join(OUT, "predictions.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"predictions.jsonl written: {len(out)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["mistral", "qwen4b", "finalize"])
    a = ap.parse_args()
    {"mistral": stage_mistral, "qwen4b": stage_qwen4b,
     "finalize": stage_finalize}[a.stage]()
