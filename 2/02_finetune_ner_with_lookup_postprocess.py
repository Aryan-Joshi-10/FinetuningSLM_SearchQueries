"""
JioTV+ NER — Fine-tune + Entity Lookup Post-processing
=======================================================
Same QLoRA fine-tuning pipeline as 01_finetune_ner.py, plus a
post-processing layer that uses the rule-based lookup tables in
entity_lookup_pipeline.py to:

  1. Canonicalize model-predicted entity_name values (e.g. akki → Akshay Kumar)
  2. Validate entity types and closed-set content_type values
  3. Drop hallucinated / unknown entities the model invents
  4. Recover missed entities via dictionary lookup (ensemble with fine-tuned model)
  5. Handle typos on augmented queries using fuzzy matching

This mirrors how the 3k augmented dataset was auto-annotated, but applied as a
post-training correction step at inference time.
"""

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

print("Active kernel Python:", sys.executable)
if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print("\n⚠️  Wrong kernel — packages will install into a different environment.")
    print("   Select kernel: Python 3.12 (JioTV NER)")
    print(f"   Expected: {VENV_PYTHON}\n")

PKGS = [
    "torch", "numpy", "ipykernel",
    "transformers>=4.51.0", "datasets>=3.0.0", "peft>=0.14.0",
    "trl>=0.15.0", "accelerate>=1.2.0", "sentencepiece>=0.2.0",
    "protobuf>=5.0.0", "scikit-learn>=1.5.0",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PKGS])

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer
print("All dependencies installed and importable ✓")


# ---------------------------------------------------------------------------
# 2. Configuration & imports
# ---------------------------------------------------------------------------
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Rule-based lookup tables & helpers
from entity_lookup_pipeline import (
    find_entities,
    canonical_entity_name,
    is_known_entity_type,
    is_valid_content_type_value,
    INTENT_PHRASES,
)

ANNOTATIONS_PATH = PROJECT_ROOT / "files" / "annotations.jsonl"
AUGMENTED_PATH = PROJECT_ROOT / "Augmented" / "augmentedQ.jsonl"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gemma-ner-lora-lookup"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/gemma-3-1b-it"
# MODEL_ID = "google/gemma-3-270m-it"

HF_TOKEN = os.environ.get("HF_TOKEN")
SEED = 42
random.seed(SEED)

SYSTEM_PROMPT = """You are a Named Entity Recognition system for JioTV+ search queries.
Extract ONLY searchable entities from the query. Do NOT tag user-intent words
(e.g. recommend, review, best, top, suggest, download, rating, latest, new).

Entity types (closed set):
- language: Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi, Gujarati, Bhojpuri
- content_type: movies | shows | live_channels | music_videos | videos | trailers
- genre: Comedy, Drama, Action, Thriller, Horror, Romance, Sci-Fi, etc.
- starcast: actor names
- director: director names
- singer: singer names
- musicDirector: music director names
- title: movie/show titles

Return a JSON array. Each item must have: keyword (exact substring from query),
entity_type, entity_name (canonical display form). Return [] if no entities.
Output ONLY valid JSON, no markdown fences or explanation."""


# ---------------------------------------------------------------------------
# 3. Data splits (reuse or rebuild)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_splits(annotations_path, augmented_path, out_dir, seed=42):
    rng = random.Random(seed)
    a, b = load_jsonl(annotations_path), load_jsonl(augmented_path)
    for x in a:
        x["_source"] = "5k"
    for x in b:
        x["_source"] = "3k_augmented"
    seen, pool = set(), []
    for x in a + b:
        if x["search_query"] not in seen:
            seen.add(x["search_query"])
            pool.append(x)
    buckets = defaultdict(list)
    for x in pool:
        has_lang = any(e["entity_type"] == "language" for e in x["Output"])
        has_dir = any(e["entity_type"] == "director" for e in x["Output"])
        buckets[(x["_source"], has_lang, has_dir)].append(x)
    train, val, test = [], [], []
    for rows in buckets.values():
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * 0.10))
        n_test = max(1, round(len(rows) * 0.10))
        val.extend(rows[:n_val])
        test.extend(rows[n_val:n_val + n_test])
        train.extend(rows[n_val + n_test:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(
                    {"search_query": r["search_query"], "Output": r["Output"]},
                    ensure_ascii=False,
                ) + "\n")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


if (SPLITS_DIR / "train.jsonl").exists():
    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")
    test_rows = load_jsonl(SPLITS_DIR / "test.jsonl")
    print(f"Loaded existing splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
else:
    train_rows, val_rows, test_rows = build_splits(ANNOTATIONS_PATH, AUGMENTED_PATH, SPLITS_DIR)


# ---------------------------------------------------------------------------
# 4. Post-processing pipeline
# ---------------------------------------------------------------------------
# Core addition over 01_finetune_ner.py — combines model predictions with
# the lookup tables from entity_lookup_pipeline.py.

INTENT_WORDS = {
    w.strip().lower()
    for phrase in INTENT_PHRASES
    for w in phrase.split()
    if w.strip()
}


def keyword_in_query(keyword: str, query: str) -> bool:
    """Check keyword is a real substring of the query (case-insensitive)."""
    return keyword.lower() in query.lower()


def is_intent_only_entity(entity: dict) -> bool:
    """Drop entities whose keyword is purely an intent word."""
    kw = entity.get("keyword", "").lower().strip()
    return kw in INTENT_WORDS


def postprocess_model_entities(
    query: str,
    model_entities: list[dict],
    *,
    use_fuzzy: bool = True,
    merge_lookup: bool = True,
    lookup_priority: str = "model_first",
) -> list[dict]:
    """
    Clean & enrich fine-tuned model output using entity_lookup_pipeline.

    Steps:
      1. Filter unknown entity types
      2. Drop intent-only keywords (review, recommend, best, ...)
      3. Verify keyword appears in query
      4. Canonicalize entity_name via lookup tables
      5. Validate content_type closed-set values
      6. Optionally merge with rule-based find_entities() to recover misses

    lookup_priority:
      - "model_first":  keep model spans, add lookup entities not overlapping
      - "lookup_first": prefer lookup spans on conflicts
    """
    cleaned = []
    seen_spans = set()

    for ent in model_entities:
        etype = ent.get("entity_type", "")
        keyword = ent.get("keyword", "")
        ename = ent.get("entity_name", "")

        if not is_known_entity_type(etype):
            continue
        if is_intent_only_entity(ent):
            continue
        if not keyword_in_query(keyword, query):
            continue

        canonical = canonical_entity_name(etype, ename, use_fuzzy=use_fuzzy)
        if canonical is None:
            # try canonicalizing from keyword instead (model may swap fields)
            canonical = canonical_entity_name(etype, keyword, use_fuzzy=use_fuzzy)
        if canonical is None:
            continue
        if etype == "content_type" and not is_valid_content_type_value(canonical):
            continue

        span_key = (keyword.lower(), etype)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        cleaned.append({
            "keyword": keyword,
            "entity_type": etype,
            "entity_name": canonical,
        })

    if not merge_lookup:
        return cleaned

    lookup_entities = find_entities(query, use_fuzzy=use_fuzzy)

    if lookup_priority == "lookup_first":
        # Lookup wins: start with lookup, fill gaps from model
        merged = list(lookup_entities)
        lookup_keys = {(e["keyword"].lower(), e["entity_type"]) for e in lookup_entities}
        for ent in cleaned:
            key = (ent["keyword"].lower(), ent["entity_type"])
            if key not in lookup_keys:
                merged.append(ent)
        return merged

    # model_first (default): model spans + non-overlapping lookup recovery
    merged = list(cleaned)
    model_keywords = {e["keyword"].lower() for e in cleaned}
    for ent in lookup_entities:
        kw = ent["keyword"].lower()
        # add lookup entity if its keyword span doesn't overlap any model keyword
        overlaps = any(kw in mk or mk in kw for mk in model_keywords)
        if not overlaps:
            merged.append(ent)
            model_keywords.add(kw)
    return merged


def annotate_with_postprocess(
    query: str,
    model_entities: list[dict],
    **kwargs,
) -> dict:
    output = postprocess_model_entities(query, model_entities, **kwargs)
    return {"search_query": query, "Output": output}


# Smoke test post-processing WITHOUT a trained model (simulated bad predictions)
simulated_model_output = [
    {"keyword": "vijay", "entity_type": "starcast", "entity_name": "vijay"},
    {"keyword": "film", "entity_type": "content_type", "entity_name": "film"},
    {"keyword": "scam 1992", "entity_type": "title", "entity_name": "scam 1992"},
    {"keyword": "review", "entity_type": "genre", "entity_name": "Review"},  # should be dropped
]
print(json.dumps(
    annotate_with_postprocess("vijay film scam 1992 review", simulated_model_output),
    indent=2, ensure_ascii=False,
))

# Typo recovery via lookup fuzzy match
typo_model_output = [{"keyword": "12t hfail", "entity_type": "title", "entity_name": "12t hfail"}]
print(json.dumps(
    annotate_with_postprocess("12t hfail review", typo_model_output, use_fuzzy=True),
    indent=2, ensure_ascii=False,
))


# ---------------------------------------------------------------------------
# 5. Training (same as 01_finetune_ner.py)
# ---------------------------------------------------------------------------

def format_target(output):
    return json.dumps([
        {"keyword": e["keyword"], "entity_type": e["entity_type"], "entity_name": e["entity_name"]}
        for e in output
    ], ensure_ascii=False)


def build_user_message(query):
    return f"Extract entities from this JioTV+ search query:\n{query}"


def rows_to_hf_dataset(rows, tokenizer):
    texts = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(row["search_query"])},
            {"role": "assistant", "content": format_target(row["Output"])},
        ]
        texts.append({
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        })
    return Dataset.from_list(texts)


import platform
USE_QLORA = platform.system() != "Darwin"
load_kwargs = {"token": HF_TOKEN, "device_map": "auto"}

if USE_QLORA:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model = prepare_model_for_kbit_training(model)
else:
    load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

train_ds = rows_to_hf_dataset(train_rows, tokenizer)
val_ds = rows_to_hf_dataset(val_rows, tokenizer)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        max_length=512,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=SEED,
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))


# ---------------------------------------------------------------------------
# 6. End-to-end inference: model → post-process
# ---------------------------------------------------------------------------

def parse_model_json(raw: str) -> list[dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group())
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


@torch.inference_mode()
def predict_raw(query: str, model, tokenizer, max_new_tokens=256) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(query)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_model_json(raw)


def predict_with_lookup(
    query: str,
    model,
    tokenizer,
    *,
    use_fuzzy: bool = True,
    merge_lookup: bool = True,
    lookup_priority: str = "model_first",
) -> dict:
    raw_entities = predict_raw(query, model, tokenizer)
    result = annotate_with_postprocess(
        query, raw_entities,
        use_fuzzy=use_fuzzy,
        merge_lookup=merge_lookup,
        lookup_priority=lookup_priority,
    )
    result["raw_model_output"] = raw_entities
    return result


# ---------------------------------------------------------------------------
# 7. Compare: model-only vs model + lookup post-processing
# ---------------------------------------------------------------------------

def entity_tuple(e):
    return (
        e.get("keyword", "").lower(),
        e.get("entity_type", "").lower(),
        e.get("entity_name", "").lower(),
    )


def evaluate(mode: str, rows, model, tokenizer, max_samples=50):
    subset = rows[:max_samples]
    tp = fp = fn = exact = 0
    for row in subset:
        q = row["search_query"]
        if mode == "model_only":
            pred = predict_raw(q, model, tokenizer)
        else:
            pred = predict_with_lookup(q, model, tokenizer)["Output"]
        gold = {entity_tuple(e) for e in row["Output"]}
        pred_set = {entity_tuple(e) for e in pred}
        tp += len(gold & pred_set)
        fp += len(pred_set - gold)
        fn += len(gold - pred_set)
        exact += int(gold == pred_set)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return {
        "mode": mode, "n": len(subset),
        "precision": p, "recall": r, "f1": f1,
        "exact_match": exact / len(subset),
    }


for mode in ["model_only", "model_plus_lookup"]:
    print(json.dumps(evaluate(mode, test_rows, model, tokenizer, max_samples=50), indent=2))


# ---------------------------------------------------------------------------
# 8. Demo queries — side-by-side comparison
# ---------------------------------------------------------------------------
demo_queries = [
    "vijay film scam 1992",
    "marathi comedy web series",
    "jawan review",
    "recommend a sci-fi movie",
    "best of fahadh",
    "12t hfail review",
    "barhmastra review",
    "best dark comedy movies like Beast",
]

for q in demo_queries:
    raw = predict_raw(q, model, tokenizer)
    final = annotate_with_postprocess(q, raw, use_fuzzy=True, merge_lookup=True)
    lookup_only = {"search_query": q, "Output": find_entities(q, use_fuzzy=True)}
    print(f"=== {q} ===")
    print("Model raw:", json.dumps(raw, ensure_ascii=False))
    print("After post-process:", json.dumps(final["Output"], ensure_ascii=False))
    print("Lookup-only:", json.dumps(lookup_only["Output"], ensure_ascii=False))
    print()


# ---------------------------------------------------------------------------
# 9. Batch inference on augmented queries
# ---------------------------------------------------------------------------
# Run post-processed inference on the full augmented set and compare to gold.

augmented_rows = load_jsonl(AUGMENTED_PATH)
predictions = []

for row in augmented_rows[:100]:  # increase to 3000 for full run
    q = row["search_query"]
    pred = predict_with_lookup(q, model, tokenizer)
    predictions.append({
        "search_query": q,
        "gold": row["Output"],
        "predicted": pred["Output"],
    })

out_path = OUTPUT_DIR / "augmented_predictions_sample.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Wrote {len(predictions)} predictions to {out_path}")

# Show a few mismatches
mismatches = [
    p for p in predictions
    if {entity_tuple(e) for e in p["gold"]} != {entity_tuple(e) for e in p["predicted"]}
]
print(f"Mismatches in sample: {len(mismatches)} / {len(predictions)}")
if mismatches:
    print(json.dumps(mismatches[0], indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 10. Production inference function
# ---------------------------------------------------------------------------

def jiotv_ner_pipeline(
    query: str,
    model,
    tokenizer,
    *,
    use_fuzzy: bool = True,
    fallback_to_lookup_only: bool = True,
) -> dict:
    """
    Production-ready pipeline:
      1. Fine-tuned model predicts entities
      2. Post-process with lookup tables
      3. If model returns nothing, optionally fall back to pure lookup
    """
    result = predict_with_lookup(
        query, model, tokenizer,
        use_fuzzy=use_fuzzy,
        merge_lookup=True,
        lookup_priority="model_first",
    )
    if fallback_to_lookup_only and not result["Output"]:
        result["Output"] = find_entities(query, use_fuzzy=use_fuzzy)
        result["fallback"] = "lookup_only"
    return result


# Example
for q in ["vijay film scam 1992", "12t hfail review"]:
    print(json.dumps(jiotv_ner_pipeline(q, model, tokenizer), indent=2, ensure_ascii=False))
    print()
