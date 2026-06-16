"""
JioTV+ Search Query NER — Fine-tune Gemma (≤1B) with QLoRA
===========================================================
Fine-tune a small language model (SLM) to extract structured entities
from JioTV+ search queries.

Entity types (closed set):
    language, content_type, genre, starcast, director, singer, musicDirector, title

Datasets:
    files/annotations.jsonl       — 5k original annotated queries
    Augmented/augmentedQ.jsonl    — 3k typo/noisy augmented queries

Default model: google/gemma-3-1b-it (1B params).
Switch to google/gemma-3-270m-it for an even smaller SLM.

GPU required for training (≥8 GB VRAM with 4-bit QLoRA).
Accept the Gemma license on Hugging Face and set HF_TOKEN before running.
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
# 2. Configuration
# ---------------------------------------------------------------------------
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(".").resolve()
ANNOTATIONS_PATH = PROJECT_ROOT / "files" / "annotations.jsonl"
AUGMENTED_PATH = PROJECT_ROOT / "Augmented" / "augmentedQ.jsonl"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gemma-ner-lora"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model: pick one SLM (both ≤1B) ---
MODEL_ID = "google/gemma-3-1b-it"          # 1B — recommended
# MODEL_ID = "google/gemma-3-270m-it"      # 270M — faster, less accurate

HF_TOKEN = os.environ.get("HF_TOKEN")      # export HF_TOKEN=hf_...
SEED = 42
random.seed(SEED)

ENTITY_TYPES = [
    "language", "content_type", "genre", "starcast",
    "director", "singer", "musicDirector", "title",
]
CONTENT_TYPE_VALUES = {
    "movies", "shows", "live_channels", "music_videos", "videos", "trailers",
}

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
# 3. Build train / val / test splits
# ---------------------------------------------------------------------------
# Combines both datasets, deduplicates queries, and stratifies by source +
# rare entity types (language, director).

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_splits(
    annotations_path: Path,
    augmented_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    a = load_jsonl(annotations_path)
    b = load_jsonl(augmented_path)
    for x in a:
        x["_source"] = "5k"
    for x in b:
        x["_source"] = "3k_augmented"

    seen, pool = set(), []
    for x in a + b:
        q = x["search_query"]
        if q in seen:
            continue
        seen.add(q)
        pool.append(x)

    def strat_key(x):
        has_lang = any(e["entity_type"] == "language" for e in x["Output"])
        has_dir = any(e["entity_type"] == "director" for e in x["Output"])
        return (x["_source"], has_lang, has_dir)

    buckets = defaultdict(list)
    for x in pool:
        buckets[strat_key(x)].append(x)

    train, val = [], []

    for rows in buckets.values():
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, round(n * 0.10))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                out = {"search_query": r["search_query"], "Output": r["Output"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    dump(train, out_dir / "train.jsonl")
    dump(val, out_dir / "val.jsonl")
    print(f"Pool (deduped): {len(pool)} | train: {len(train)} | val: {len(val)}")
    return train, val


train_rows, val_rows = build_splits(
    ANNOTATIONS_PATH, AUGMENTED_PATH, SPLITS_DIR, seed=SEED
)

# Quick sanity check
print("Sample:", json.dumps(train_rows[0], indent=2, ensure_ascii=False))

total = len(train_rows) + len(val_rows)
print(f"Train %: {100 * len(train_rows) / total:.2f}")
print(f"Val %: {100 * len(val_rows) / total:.2f}")


# ---------------------------------------------------------------------------
# 4. Format examples for instruction fine-tuning
# ---------------------------------------------------------------------------

def format_target(output: list[dict]) -> str:
    """Compact JSON target the model learns to generate."""
    clean = [
        {
            "keyword": e["keyword"],
            "entity_type": e["entity_type"],
            "entity_name": e["entity_name"],
        }
        for e in output
    ]
    return json.dumps(clean, ensure_ascii=False)


def build_user_message(query: str) -> str:
    return f"Extract entities from this JioTV+ search query:\n{query}"


def row_to_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(row["search_query"])},
        {"role": "assistant", "content": format_target(row["Output"])},
    ]


def rows_to_hf_dataset(rows: list[dict], tokenizer) -> Dataset:
    texts = []
    for row in rows:
        messages = row_to_messages(row)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})
    return Dataset.from_list(texts)


# Preview one formatted example (tokenizer loaded in next section)
preview_row = train_rows[0]
print("Query:", preview_row["search_query"])
print("Target JSON:", format_target(preview_row["Output"]))


# ---------------------------------------------------------------------------
# 5. Load tokenizer & model (4-bit QLoRA)
# ---------------------------------------------------------------------------
import platform

USE_QLORA = platform.system() != "Darwin"  # bitsandbytes 4-bit needs CUDA (Linux/Windows+GPU)

load_kwargs = {"token": HF_TOKEN, "device_map": "auto"}
if USE_QLORA:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model = prepare_model_for_kbit_training(model)
    print("Loaded with 4-bit QLoRA (CUDA)")
else:
    dtype = torch.float16
    if torch.backends.mps.is_available():
        dtype = torch.float16
        print("macOS detected — using LoRA in float16 on MPS (no bitsandbytes)")
    else:
        print("macOS/CPU detected — using LoRA in float16")
    load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_ds = rows_to_hf_dataset(train_rows, tokenizer)
val_ds = rows_to_hf_dataset(val_rows, tokenizer)
print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")
print("\n--- Formatted training example (first 800 chars) ---")
print(train_ds[0]["text"][:800])


# ---------------------------------------------------------------------------
# 6. Train with SFTTrainer
# ---------------------------------------------------------------------------
training_args = SFTConfig(
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
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
print(f"Saved adapter + tokenizer to {OUTPUT_DIR / 'final'}")


# ---------------------------------------------------------------------------
# 7. Inference helper
# ---------------------------------------------------------------------------

def parse_model_json(raw: str) -> list[dict]:
    """Parse model output into a list of entity dicts."""
    text = raw.strip()
    # strip markdown fences if model adds them
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # grab first JSON array
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


@torch.inference_mode()
def predict_entities(
    query: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(query)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    entities = parse_model_json(raw)
    return {"search_query": query, "Output": entities, "raw_model_output": raw}


# ---------------------------------------------------------------------------
# 8. Evaluate on validation set
# ---------------------------------------------------------------------------

def entity_tuple(e: dict) -> tuple:
    """Normalize for comparison (case-insensitive keyword + type + name)."""
    return (
        e.get("keyword", "").lower().strip(),
        e.get("entity_type", "").lower().strip(),
        e.get("entity_name", "").lower().strip(),
    )


def evaluate_rows(rows: list[dict], model, tokenizer, max_samples: int | None = None) -> dict:
    subset = rows[:max_samples] if max_samples else rows
    tp = fp = fn = 0
    exact_match = 0

    for row in subset:
        pred = predict_entities(row["search_query"], model, tokenizer)["Output"]
        gold_set = {entity_tuple(e) for e in row["Output"]}
        pred_set = {entity_tuple(e) for e in pred}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        if gold_set == pred_set:
            exact_match += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(subset),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match / len(subset),
    }


# Quick smoke eval on 50 queries
metrics = evaluate_rows(val_rows, model, tokenizer, max_samples=50)
print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# 9. Demo on example queries
# ---------------------------------------------------------------------------
demo_queries = [
    "vijay film scam 1992",
    "marathi comedy web series",
    "jawan review",
    "recommend a sci-fi movie",
    "best of fahadh",
    "12t hfail review",
]

for q in demo_queries:
    result = predict_entities(q, model, tokenizer)
    print(json.dumps(
        {"search_query": result["search_query"], "Output": result["Output"]},
        indent=2, ensure_ascii=False,
    ))
    print()


# ---------------------------------------------------------------------------
# 10. Merge LoRA & export (optional, for deployment)
# ---------------------------------------------------------------------------
# Run this only if you need a standalone merged checkpoint (larger disk footprint).

# from peft import PeftModel

# base = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", token=HF_TOKEN
# )
# merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR / "final"))
# merged = merged.merge_and_unload()
# merged.save_pretrained(str(OUTPUT_DIR / "merged"))
# tokenizer.save_pretrained(str(OUTPUT_DIR / "merged"))
# print("Merged model saved.")
