import argparse
import csv
import heapq
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


PROMPT_KEYS = ("prompt", "caption", "text", "description")
TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "with", "without",
    "and", "or", "but", "by", "is", "are", "was", "were", "be", "being", "been",
    "as", "into", "over", "under", "above", "below", "near", "next", "beside", "around",
    "this", "that", "these", "those", "some", "any", "very", "really",
}

COLOR_WORDS = {
    "black", "white", "red", "green", "blue", "yellow", "purple", "pink", "orange",
    "brown", "gray", "grey", "cyan", "magenta", "gold", "silver", "violet", "beige",
    "turquoise", "rainbow",
}

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "single", "double", "triple", "several", "many", "few", "couple", "pair",
}

SPATIAL_WORDS = {
    "left", "right", "top", "bottom", "front", "behind", "between", "inside", "outside",
    "under", "underneath", "above", "below", "near", "next", "beside", "around", "over",
    "across", "through", "against",
}

STYLE_WORDS = {
    "photo", "photograph", "painting", "drawing", "illustration", "sketch", "render",
    "cartoon", "anime", "oil", "watercolor", "macro", "portrait", "landscape", "poster",
    "cinematic", "studio", "lighting", "digital", "art", "realistic", "surreal",
}

ACTION_WORDS = {
    "eating", "playing", "driving", "riding", "holding", "wearing", "standing", "sitting",
    "walking", "running", "jumping", "flying", "swimming", "looking", "making", "washing",
    "cut", "arranged",
}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def first_existing(row, names):
    for name in names:
        if name and name in row and row[name] is not None:
            return row[name]
    return None


def extract_prompts_from_json(obj, prompt_column=None):
    prompt_keys = tuple([prompt_column] if prompt_column else []) + PROMPT_KEYS

    if isinstance(obj, list):
        prompts = []
        for item in obj:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                value = first_existing(item, prompt_keys)
                if value is not None:
                    prompts.append(str(value))
        return prompts

    if isinstance(obj, dict):
        for key in (prompt_column, "prompts", "prompt", "captions", "caption", "texts", "text"):
            if key and key in obj:
                value = obj[key]
                if isinstance(value, list):
                    return [str(item) for item in value if item is not None]
                if isinstance(value, str):
                    return [value]

        if "item_idx" in obj and isinstance(obj["item_idx"], dict):
            return [str(prompt) for prompt in obj["item_idx"].keys()]

        prompts = []
        for value in obj.values():
            if isinstance(value, dict):
                item_prompt = first_existing(value, prompt_keys)
                if item_prompt is not None:
                    prompts.append(str(item_prompt))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item_prompt = first_existing(item, prompt_keys)
                        if item_prompt is not None:
                            prompts.append(str(item_prompt))
        if prompts:
            return prompts

        return [str(prompt) for prompt in obj.keys()]

    raise ValueError("Unsupported JSON structure")


def load_prompts_from_dataset(path, split="train", prompt_column=None, max_prompts=-1):
    import datasets

    dataset = datasets.load_from_disk(path)
    if isinstance(dataset, datasets.DatasetDict):
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f'Split "{split}" was not found. Available splits: {available}')
        dataset = dataset[split]

    if prompt_column is None:
        for key in PROMPT_KEYS:
            if key in dataset.column_names:
                prompt_column = key
                break
    if prompt_column is None or prompt_column not in dataset.column_names:
        raise ValueError(f"Prompt column was not found. Available columns: {dataset.column_names}")

    limit = len(dataset) if max_prompts is None or max_prompts < 0 else min(max_prompts, len(dataset))
    return [str(dataset[i][prompt_column]) for i in range(limit)]


def load_prompts(path, split="train", prompt_column=None, max_prompts=-1):
    path = Path(path)
    if path.is_dir():
        return load_prompts_from_dataset(str(path), split=split, prompt_column=prompt_column, max_prompts=max_prompts)

    suffix = path.suffix.lower()
    if suffix == ".json":
        prompts = extract_prompts_from_json(read_json(path), prompt_column=prompt_column)
    elif suffix == ".jsonl":
        prompts = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    value = first_existing(item, tuple([prompt_column] if prompt_column else []) + PROMPT_KEYS)
                    if value is not None:
                        prompts.append(str(value))
    elif suffix in {".txt", ".prompts"}:
        with open(path, "r", encoding="utf-8-sig") as f:
            prompts = [line.rstrip("\n") for line in f]
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if prompt_column is None:
                for key in PROMPT_KEYS:
                    if key in reader.fieldnames:
                        prompt_column = key
                        break
            if prompt_column is None:
                raise ValueError(f"Prompt column was not found. Available columns: {reader.fieldnames}")
            prompts = [row[prompt_column] for row in reader if row.get(prompt_column) is not None]
    else:
        raise ValueError(f"Unsupported prompt file extension: {suffix}")

    prompts = [str(prompt).strip() for prompt in prompts]
    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def tokenize(text):
    return TOKEN_RE.findall(text.casefold())


def normalize_prompt(text):
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_tokens(text):
    return [token for token in tokenize(text) if token not in STOPWORDS]


def canonical_bow_key(text):
    return " ".join(sorted(content_tokens(text)))


def char_ngrams(text, n=3):
    text = normalize_prompt(text).replace(" ", "_")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def structural_similarity(cache, i, j):
    token_sim = jaccard(cache["token_sets"][i], cache["token_sets"][j])
    char_sim = jaccard(cache["char_sets"][i], cache["char_sets"][j])
    bow_same = 1.0 if cache["bow_keys"][i] and cache["bow_keys"][i] == cache["bow_keys"][j] else 0.0
    return 0.55 * token_sim + 0.35 * char_sim + 0.10 * bow_same, token_sim, char_sim, bow_same


def build_similarity_cache(prompts):
    return {
        "normalized": [normalize_prompt(prompt) for prompt in prompts],
        "bow_keys": [canonical_bow_key(prompt) for prompt in prompts],
        "token_sets": [set(content_tokens(prompt)) for prompt in prompts],
        "char_sets": [char_ngrams(prompt) for prompt in prompts],
    }


def describe_numbers(values):
    values = np.asarray([value for value in values if value is not None and not math.isnan(float(value))], dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def prompt_quality_metrics(prompt):
    stripped = prompt.strip()
    tokens = tokenize(stripped)
    non_space = max(1, len(re.sub(r"\s+", "", stripped)))
    alnum_ratio = sum(ch.isalnum() for ch in stripped) / non_space
    unusual_ratio = sum((not ch.isalnum()) and (not ch.isspace()) and ch not in ".,;:!?'-()/&" for ch in stripped) / non_space
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    adjacent_repeats = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    has_url = bool(URL_RE.search(stripped))
    has_html = bool(HTML_RE.search(stripped))
    has_replacement = "\ufffd" in stripped or "В©" in stripped
    content_count = len(content_tokens(stripped))

    score = 1.0
    if len(tokens) == 0:
        score = 0.0
    else:
        if len(tokens) < 3:
            score -= 0.30
        if len(tokens) > 50:
            score -= min(0.30, (len(tokens) - 50) / 120)
        if content_count == 0:
            score -= 0.25
        if unique_ratio < 0.55:
            score -= 0.20
        if alnum_ratio < 0.65:
            score -= 0.20
        if unusual_ratio > 0.12:
            score -= 0.20
        if has_url:
            score -= 0.25
        if has_html:
            score -= 0.25
        if has_replacement:
            score -= 0.20
        if adjacent_repeats:
            score -= min(0.20, 0.05 * adjacent_repeats)

    return {
        "quality_score": clamp(score),
        "word_count": len(tokens),
        "char_count": len(stripped),
        "content_word_count": content_count,
        "unique_token_ratio": unique_ratio,
        "alnum_ratio": alnum_ratio,
        "unusual_char_ratio": unusual_ratio,
        "adjacent_repeats": adjacent_repeats,
        "has_url": has_url,
        "has_html": has_html,
        "has_replacement_chars": has_replacement,
    }


def prompt_complexity_metrics(prompt):
    tokens = tokenize(prompt)
    token_set = set(tokens)
    content = content_tokens(prompt)
    color_count = sum(token in COLOR_WORDS for token in tokens)
    number_count = sum(token.isdigit() or token in NUMBER_WORDS for token in tokens)
    spatial_count = sum(token in SPATIAL_WORDS for token in tokens)
    style_count = sum(token in STYLE_WORDS for token in tokens)
    action_count = sum(token in ACTION_WORDS or token.endswith("ing") for token in tokens)
    separator_count = prompt.count(",") + prompt.count(";") + prompt.count(":")
    clause_count = 1 + separator_count + len(re.findall(r"\b(and|with|while|that|which)\b", prompt.casefold()))

    score = (
        0.22 * min(len(tokens) / 24, 1.0)
        + 0.18 * min(len(set(content)) / 10, 1.0)
        + 0.13 * min(color_count / 3, 1.0)
        + 0.12 * min(number_count / 2, 1.0)
        + 0.14 * min(spatial_count / 2, 1.0)
        + 0.10 * min(style_count / 2, 1.0)
        + 0.07 * min(action_count / 2, 1.0)
        + 0.04 * min(max(0, clause_count - 1) / 3, 1.0)
    )

    return {
        "complexity_score": clamp(score),
        "word_count": len(tokens),
        "content_word_count": len(content),
        "unique_content_words": len(set(content)),
        "color_count": color_count,
        "number_count": number_count,
        "spatial_relation_count": spatial_count,
        "style_word_count": style_count,
        "action_word_count": action_count,
        "clause_estimate": clause_count,
        "has_color": color_count > 0,
        "has_number": number_count > 0,
        "has_spatial_relation": spatial_count > 0,
        "has_style": style_count > 0,
        "has_action": action_count > 0,
        "has_multiple_objects_hint": len(token_set & {"and", "with"}) > 0,
    }


def prompt_template_signature(prompt):
    parts = []
    for token in tokenize(prompt):
        if token in COLOR_WORDS:
            parts.append("<color>")
        elif token.isdigit() or token in NUMBER_WORDS:
            parts.append("<number>")
        elif token in SPATIAL_WORDS:
            parts.append("<spatial>")
        elif token in STYLE_WORDS:
            parts.append("<style>")
        elif token in ACTION_WORDS or token.endswith("ing"):
            parts.append("<action>")
        elif token in STOPWORDS:
            parts.append(token)
        else:
            parts.append("<content>")
    return " ".join(parts)


def grouped_duplicates(prompts, keys, max_groups=50):
    groups = {}
    for idx, key in enumerate(keys):
        if key:
            groups.setdefault(key, []).append(idx)
    rows = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        rows.append({
            "key": key,
            "size": len(indices),
            "indices": indices[:20],
            "sample_prompts": [prompts[idx] for idx in indices[:5]],
        })
    rows.sort(key=lambda row: row["size"], reverse=True)
    return rows[:max_groups]


def sample_pair_indices(n, max_pairs=200000, seed=123):
    if n < 2:
        return []
    total_pairs = n * (n - 1) // 2
    if max_pairs < 0 or total_pairs <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < max_pairs:
        left = rng.integers(0, n, size=max_pairs)
        right = rng.integers(0, n, size=max_pairs)
        for i, j in zip(left, right):
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pairs.add((int(i), int(j)))
            if len(pairs) >= max_pairs:
                break
    return list(pairs)


def update_top(heap, score, row, limit):
    if limit <= 0:
        return
    item = (float(score), id(row), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def sorted_heap_rows(heap):
    return [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]


def compute_text_embeddings(prompts, model_name, batch_size=64, device=None, local_files_only=False, trust_remote_code=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model.eval().to(device)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


def pairwise_report(prompts, pairs, top_k=50, semantic_embeddings=None):
    cache = build_similarity_cache(prompts)
    structural_values = []
    token_values = []
    char_values = []
    bow_same_count = 0
    semantic_values = []
    top_structural = []
    top_semantic = []

    for i, j in pairs:
        structural, token_sim, char_sim, bow_same = structural_similarity(cache, i, j)
        structural_values.append(structural)
        token_values.append(token_sim)
        char_values.append(char_sim)
        bow_same_count += int(bow_same > 0)

        row = {
            "i": i,
            "j": j,
            "structural_similarity": structural,
            "token_jaccard": token_sim,
            "char_ngram_jaccard": char_sim,
            "same_bow": bool(bow_same),
            "prompt_i": prompts[i],
            "prompt_j": prompts[j],
        }
        update_top(top_structural, structural, row, top_k)

        if semantic_embeddings is not None:
            semantic = float(np.dot(semantic_embeddings[i], semantic_embeddings[j]))
            semantic_values.append(semantic)
            semantic_row = dict(row)
            semantic_row["semantic_similarity"] = semantic
            update_top(top_semantic, semantic, semantic_row, top_k)

    summary = {
        "pair_count": len(pairs),
        "structural_similarity": describe_numbers(structural_values),
        "token_jaccard": describe_numbers(token_values),
        "char_ngram_jaccard": describe_numbers(char_values),
        "same_bow_pair_ratio": bow_same_count / max(1, len(pairs)),
        "structural_homogeneity": float(np.mean(structural_values)) if structural_values else None,
        "structural_heterogeneity": 1.0 - float(np.mean(structural_values)) if structural_values else None,
    }
    if semantic_embeddings is not None:
        semantic_mean = float(np.mean(semantic_values)) if semantic_values else None
        summary["semantic_similarity"] = describe_numbers(semantic_values)
        summary["semantic_homogeneity"] = semantic_mean
        summary["semantic_heterogeneity"] = 1.0 - semantic_mean if semantic_mean is not None else None

    return summary, sorted_heap_rows(top_structural), sorted_heap_rows(top_semantic)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_dir(prompt_path):
    stem = Path(prompt_path).stem if prompt_path else "prompts"
    return Path(__file__).resolve().parent / "prompt_reports" / stem



def analyze_prompt_set(args):
    prompts = load_prompts(
        args.prompts,
        split=args.split,
        prompt_column=args.prompt_column,
        max_prompts=args.max_prompts,
    )
    if not prompts:
        raise ValueError("No prompts were loaded")

    quality = [prompt_quality_metrics(prompt) for prompt in prompts]
    complexity = [prompt_complexity_metrics(prompt) for prompt in prompts]
    normalized = [normalize_prompt(prompt) for prompt in prompts]
    bow_keys = [canonical_bow_key(prompt) for prompt in prompts]
    template_keys = [prompt_template_signature(prompt) for prompt in prompts]

    pairs = sample_pair_indices(len(prompts), max_pairs=args.max_pairs, seed=args.seed)
    semantic_embeddings = None
    semantic_error = None
    if args.semantic_model and args.semantic_model.lower() != "none":
        try:
            semantic_embeddings = compute_text_embeddings(
                prompts,
                args.semantic_model,
                batch_size=args.semantic_batch_size,
                device=args.device,
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
            )
        except Exception as exc:
            semantic_error = repr(exc)

    pair_summary, top_structural, top_semantic = pairwise_report(
        prompts,
        pairs,
        top_k=args.top_k,
        semantic_embeddings=semantic_embeddings,
    )

    normalized_counter = Counter(normalized)
    bow_counter = Counter(key for key in bow_keys if key)
    template_counter = Counter(template_keys)
    vocabulary = Counter(token for prompt in prompts for token in content_tokens(prompt))

    summary = {
        "source": str(args.prompts),
        "num_prompts": len(prompts),
        "num_empty_prompts": sum(1 for prompt in prompts if not prompt.strip()),
        "num_unique_exact": len(set(prompts)),
        "num_unique_normalized": len(normalized_counter),
        "num_unique_bow": len(bow_counter),
        "exact_duplicate_prompt_count": sum(count - 1 for count in Counter(prompts).values() if count > 1),
        "normalized_duplicate_prompt_count": sum(count - 1 for count in normalized_counter.values() if count > 1),
        "word_order_or_bow_duplicate_count": sum(count - 1 for count in bow_counter.values() if count > 1),
        "quality": {
            "quality_score": describe_numbers([item["quality_score"] for item in quality]),
            "word_count": describe_numbers([item["word_count"] for item in quality]),
            "content_word_count": describe_numbers([item["content_word_count"] for item in quality]),
            "unique_token_ratio": describe_numbers([item["unique_token_ratio"] for item in quality]),
            "url_like_count": sum(item["has_url"] for item in quality),
            "html_like_count": sum(item["has_html"] for item in quality),
            "replacement_char_count": sum(item["has_replacement_chars"] for item in quality),
        },
        "complexity": {
            "complexity_score": describe_numbers([item["complexity_score"] for item in complexity]),
            "color_prompt_ratio": float(np.mean([item["has_color"] for item in complexity])),
            "number_prompt_ratio": float(np.mean([item["has_number"] for item in complexity])),
            "spatial_prompt_ratio": float(np.mean([item["has_spatial_relation"] for item in complexity])),
            "style_prompt_ratio": float(np.mean([item["has_style"] for item in complexity])),
            "action_prompt_ratio": float(np.mean([item["has_action"] for item in complexity])),
            "multi_object_hint_ratio": float(np.mean([item["has_multiple_objects_hint"] for item in complexity])),
        },
        "pairwise": pair_summary,
        "template_families": {
            "num_unique_templates": len(template_counter),
            "largest_template_families": [
                {"template": key, "size": count}
                for key, count in template_counter.most_common(args.top_k)
            ],
        },
        "top_content_words": [
            {"token": token, "count": count}
            for token, count in vocabulary.most_common(args.top_k)
        ],
        "duplicate_groups_normalized": grouped_duplicates(prompts, normalized, max_groups=args.top_k),
        "duplicate_groups_bow": grouped_duplicates(prompts, bow_keys, max_groups=args.top_k),
        "semantic_model": args.semantic_model,
        "semantic_error": semantic_error,
    }

    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(args.prompts)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "top_structural_pairs.csv", top_structural)
    if semantic_embeddings is not None:
        write_csv(out_dir / "top_semantic_pairs.csv", top_semantic)

    print(f"Loaded prompts: {len(prompts)}")
    print(f"Report: {out_dir / 'summary.json'}")
    print(f"Top structural pairs: {out_dir / 'top_structural_pairs.csv'}")
    if semantic_embeddings is not None:
        print(f"Top semantic pairs: {out_dir / 'top_semantic_pairs.csv'}")
    elif semantic_error:
        print(f"Semantic similarity skipped: {semantic_error}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze homogeneity, diversity and similarity inside a prompt set.")
    parser.add_argument("--prompts", required=True, help="Path to json/jsonl/txt/csv prompt file or load_from_disk dataset")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt_column", default=None)
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument("--max_pairs", type=int, default=200000)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--semantic_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic_batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", type=int, default=0)
    parser.add_argument("--trust_remote_code", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    analyze_prompt_set(parse_args())
