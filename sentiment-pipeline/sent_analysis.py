import os, sys, gc, uuid, torch, asyncio
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, Dataset
from transformers import (pipeline, AutoModelForSequenceClassification, 
                          AutoTokenizer)
from huggingface_hub import login
from tqdm.auto import tqdm

HF_TOKEN = "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" 
OUTPUT_REPO = "mateiplescan/processed-financial-news-XXL-enriched" 
OUTPUT_REPO_AGGREG = "mateiplescan/processed-financial-news-XXL-enriched-agg" 
SOURCE_DATASET = "Brianferrell787/financial-news-multisource"
login(token=HF_TOKEN)

SYMBOLS = [
    "NATURAL GAS", "GOLD", "WTI CRUDE", "BRENT CRUDE", "SOYBEANS",
    "CORN", "COPPER", "SILVER", "LOW SULPHUR GAS OIL",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "JPM", "XOM", "LLY", "UNH"
]

KEYWORD_MAP = {
    "NATURAL GAS": [
        "natural gas", "lng", "natgas", "liquefied natural gas",
        "gas prices", "henry hub", "gas futures", "gas supply",
        "gas pipeline", "gas reserves", "gas production", "gas exports",
        "gas imports", "gas storage", "gas demand", "ttf gas",
    ],
    "GOLD": [
        "gold", "xau", "bullion", "gold prices", "gold futures",
        "gold reserves", "gold mining", "gold etf", "comex gold",
        "precious metal", "gold rally", "gold demand", "gold supply",
        "safe haven", "spot gold",
    ],
    "WTI CRUDE": [
        "wti", "crude oil", "west texas intermediate", "oil prices",
        "oil futures", "crude futures", "nymex crude", "oil supply",
        "oil demand", "oil production", "oil reserves", "oil exports",
        "oil imports", "opec", "barrel of oil", "shale oil",
        "oil market", "petroleum",
    ],
    "BRENT CRUDE": [
        "brent", "brent crude", "brent oil", "ice brent",
        "north sea oil", "brent futures", "brent prices",
        "global oil benchmark", "dated brent",
    ],
    "SOYBEANS": [
        "soybean", "soybeans", "soy", "soya", "soy futures",
        "soybean oil", "soybean meal", "soy crush", "cbot soy",
        "soy exports", "soy harvest", "soy crop", "soy demand",
        "oilseed",
    ],
    "CORN": [
        "corn", "maize", "corn futures", "corn prices", "corn crop",
        "corn harvest", "corn exports", "corn demand", "corn supply",
        "ethanol corn", "cbot corn", "feed grain", "grain prices",
    ],
    "COPPER": [
        "copper", "copper prices", "copper futures", "comex copper",
        "lme copper", "copper demand", "copper supply", "copper mining",
        "red metal", "copper ore", "copper exports", "dr copper",
        "copper rally", "copper production",
    ],
    "SILVER": [
        "silver", "xag", "silver prices", "silver futures",
        "silver mining", "comex silver", "spot silver", "silver etf",
        "silver demand", "industrial silver", "silver rally",
        "precious metals",
    ],
    "LOW SULPHUR GAS OIL": [
        "gas oil", "gasoil", "sulphur", "low sulphur", "lsgo",
        "ultra low sulphur", "ulsd", "diesel futures", "heating oil",
        "distillate", "ice gasoil", "gas oil futures",
    ],
    "AAPL": [
        "apple", "aapl", "iphone", "ipad", "macbook", "mac",
        "apple inc", "app store", "apple watch", "airpods",
        "tim cook", "apple vision", "ios", "macos", "apple silicon",
        "cupertino", "apple earnings", "apple revenue",
    ],
    "MSFT": [
        "microsoft", "msft", "windows", "azure", "office 365",
        "xbox", "linkedin", "teams", "satya nadella", "bing",
        "microsoft cloud", "github", "copilot", "dynamics",
        "microsoft earnings", "sharepoint", "onedrive",
    ],
    "NVDA": [
        "nvidia", "nvda", "gpu", "geforce", "cuda", "h100",
        "a100", "jensen huang", "nvidia earnings", "blackwell",
        "hopper", "nvidia chip", "ai chip", "graphics card",
        "nvidia revenue", "nvidia data center", "rtx",
    ],
    "AMZN": [
        "amazon", "amzn", "aws", "amazon web services", "prime",
        "alexa", "amazon prime", "andy jassy", "amazon fresh",
        "whole foods", "amazon earnings", "amazon revenue",
        "amazon cloud", "amazon marketplace", "fulfillment center",
        "amazon logistics",
    ],
    "GOOGL": [
        "google", "alphabet", "googl", "goog", "youtube", "gmail",
        "google cloud", "sundar pichai", "android", "google search",
        "waymo", "deepmind", "google ads", "google earnings",
        "google revenue", "pixel phone", "chrome", "google maps",
    ],
    "TSLA": [
        "tesla", "tsla", "elon musk", "electric vehicle", "ev maker",
        "model 3", "model y", "model s", "model x", "cybertruck",
        "supercharger", "tesla earnings", "tesla revenue",
        "gigafactory", "autopilot", "full self driving", "fsd",
        "powerwall", "tesla energy",
    ],
    "JPM": [
        "jpmorgan", "jp morgan", "j.p. morgan", "jpm",
        "jamie dimon", "chase bank", "jpmorgan chase",
        "jpmorgan earnings", "investment banking", "jpmorgan revenue",
        "chase credit", "jpmorgan asset management",
    ],
    "XOM": [
        "exxon", "exxonmobil", "exxon mobil",
        "darren woods", "exxon earnings", "exxon revenue",
        "exxon refinery", "exxon oil", "exxon chemical",
        "exxon upstream", "exxon downstream",
    ],
    "LLY": [
        "eli lilly", "lilly", "mounjaro", "zepbound",
        "tirzepatide", "verzenio", "jardiance", "trulicity",
        "david ricks", "lilly earnings", "lilly revenue",
        "lilly diabetes", "lilly oncology", "weight loss drug",
        "glp-1", "obesity drug",
    ],
    "UNH": [
        "unitedhealth", "united health", "unitedhealthcare",
        "optum", "andrew witty", "unitedhealth earnings",
        "unitedhealth revenue", "health insurance", "managed care",
        "unitedhealth group", "optum rx", "optum health",
    ],
}

import re

# Pre-compile the regex patterns for each symbol for word-boundary matching
_SEP = r'(?:(?<=^)|(?<=[\s\.,;:!\?\-\(\)\[\]\"\'\/]))'
_SEP_END = r'(?=[\s\.,;:!\?\-\(\)\[\]\"\'\/]|$)'

COMPILED_KEYWORDS = {
    sym: re.compile(
        r'(?<![a-zA-Z0-9])(?:' + '|'.join(map(re.escape, kws)) + r')(?![a-zA-Z0-9])',
        re.IGNORECASE
    )
    for sym, kws in KEYWORD_MAP.items()
}

def fast_keyword_labels(text: str) -> list[str]:
    return [sym for sym, pattern in COMPILED_KEYWORDS.items() if pattern.search(text)]

if not torch.cuda.is_available():
    print("❌ No GPU found."); sys.exit(1)

print("🚀 Loading models with torch.compile() for H100...")

def load_pipe(model_name, task):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipeline(task, model=model, tokenizer=tok, truncation=True, max_length=512)

sent_pipe = load_pipe("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis", "sentiment-analysis")

def process_batch(batch):
    texts = [str(t) for t in batch['text']]
    # Provide a placeholder string if a text is completely empty to avoid pipeline ValueErrors
    texts = [t if t.strip() else " " for t in texts]

    keyword_results = [fast_keyword_labels(t) for t in texts]
    
    # Identify which texts have keyword hits
    valid_indices = [i for i, kws in enumerate(keyword_results) if len(kws) > 0]
    valid_texts = [texts[i] for i in valid_indices]
    
    # Only run sentiment pipeline on texts that returned a keyword hit
    if valid_texts:
        sent_results = sent_pipe(valid_texts, batch_size=64, top_k=None)
    else:
        sent_results = []

    # Map the sentiment scores back to their original batch positions
    sent_dict = {}
    for idx, res in zip(valid_indices, sent_results):
        s = {r['label'].lower(): r['score'] for r in res}
        sent_dict[idx] = round(s.get('positive', 0) - s.get('negative', 0), 4)

    final_sent, final_labels = [], []
    for i in range(len(texts)):
        final_sent.append(sent_dict.get(i, 0.0))  # defaults to 0.0 for dropped rows
        final_labels.append(keyword_results[i])

    return {
        "sentiment_score": final_sent,
        "symbol_label":    final_labels,
        "unique_id":       [str(uuid.uuid4()) for _ in range(len(texts))],
    }

# --- Main loop ---
print(f"📂 Connecting to {SOURCE_DATASET}...")
hf_stream = load_dataset(SOURCE_DATASET, split="train", streaming=True)

buffer, shard_count = [], 0
BUFFER_SIZE = 50000  # Larger buffers = fewer Hub pushes (push overhead is real)

import random
SAMPLE_RATE = 1

print("⚙️  Starting optimized loop (100% sample)...")
for row in tqdm(hf_stream):
    if random.random() > SAMPLE_RATE:
        continue
    buffer.append(row)
    if len(buffer) >= BUFFER_SIZE:
        print(f"\n📦 Processing shard {shard_count}...")
        temp_ds = Dataset.from_list(buffer)
        processed_ds = temp_ds.map(
            process_batch,
            batched=True,
            batch_size=4096,       # Larger map batch = fewer Python→Rust round trips
        )
        filtered_ds = processed_ds.filter(lambda x: len(x["symbol_label"]) > 0)
        
        if len(filtered_ds) > 0:
            filtered_ds.push_to_hub(OUTPUT_REPO, private=True, split=f"shard_{shard_count}")
        del temp_ds, processed_ds, filtered_ds
        buffer = []
        shard_count += 1
        gc.collect()
        torch.cuda.empty_cache()

if buffer:
    temp_ds = Dataset.from_list(buffer)
    cols_to_remove = [c for c in ["extra_fields"] if c in temp_ds.column_names]
    final_ds = temp_ds.map(process_batch, batched=True, batch_size=4096,
                            remove_columns=cols_to_remove)
    filtered_final_ds = final_ds.filter(lambda x: len(x["symbol_label"]) > 0)
    if len(filtered_final_ds) > 0:
        filtered_final_ds.push_to_hub(OUTPUT_REPO, private=False, split=f"shard_{shard_count}")

print(f"🏁 Done! → https://huggingface.co/datasets/{OUTPUT_REPO}")