#!/usr/bin/env python3
"""Shared library for the LM-KBC 2026 system: prompts, few-shot exemplars,
parsers, decision-layer utilities, and the HF sampling/probing backend.

Closed-book: all facts come from the model's parameters. The system runs on a
single backbone, meta-models/Muse-Glimmer-30B (29.6B parameters, inside the
32B budget on its own). The pipeline lives in three sibling scripts:

  muse_elicit.py       sampling channels + one-token probe stages (GPU)
  muse_channels.py     auxiliary channels, exchange/capterm/border-freq stages
  build_submission.py  deterministic fusion + decision rules -> predictions

Outputs land in <repo>/outputs/: per-relation checkpoints (resumable) and the
final predictions.jsonl in test-set order. Sampling channels use temperature
sampling and are stochastic run-to-run; scores reproduce within about one
macro-F1 point. Everything downstream of the checkpoints is deterministic.
"""
import argparse, json, math, os, random, re, unicodedata
from collections import Counter, defaultdict

ROOT = os.environ.get("AKBC_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "dataset2026", "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


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
        """Summed logprob of continuation given prefix (teacher-forced).

        The float32 full-vocab logits of a batch are the peak allocation of
        the whole pipeline ([batch, seq, 202k] floats). On a sharded 2x32GB
        split the head GPU has little headroom left, so on out-of-memory the
        batch halves and the chunk retries rather than killing the stage."""
        torch = self.torch
        out = []
        i = 0
        while i < len(pairs):
            chunk = pairs[i:i + batch]
            try:
                out.extend(self._seq_logprob_chunk(chunk))
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                continue
            i += len(chunk)
        return out

    def _seq_logprob_chunk(self, chunk):
        torch = self.torch
        out = []
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
