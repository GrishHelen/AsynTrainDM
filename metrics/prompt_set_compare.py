import argparse
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prompt_set_analysis import (
    build_similarity_cache,
    canonical_bow_key,
    compute_text_embeddings,
    content_tokens,
    describe_numbers,
    jaccard,
    load_prompts,
    normalize_prompt,
    pairwise_report,
    prompt_complexity_metrics,
    prompt_quality_metrics,
    structural_similarity,
    write_csv,
    write_json,
)


def default_output_dir(path_a, path_b):
    stem_a = Path(path_a).stem if path_a else "a"
    stem_b = Path(path_b).stem if path_b else "b"
    return Path(__file__).resolve().parent / "prompt_comparisons" / f"{stem_a}__{stem_b}"


def sample_cross_pair_indices(n_a, n_b, max_pairs=300000, seed=123):
    if n_a == 0 or n_b == 0:
        return []
    total_pairs = n_a * n_b
    if max_pairs < 0 or total_pairs <= max_pairs:
        return [(i, j) for i in range(n_a) for j in range(n_b)]

    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < max_pairs:
        left = rng.integers(0, n_a, size=max_pairs)
        right = rng.integers(0, n_b, size=max_pairs)
        for i, j in zip(left, right):
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


def prompt_set_basic_summary(prompts):
    quality = [prompt_quality_metrics(prompt) for prompt in prompts]
    complexity = [prompt_complexity_metrics(prompt) for prompt in prompts]
    normalized = [normalize_prompt(prompt) for prompt in prompts]
    bow_keys = [canonical_bow_key(prompt) for prompt in prompts]
    vocab = Counter(token for prompt in prompts for token in content_tokens(prompt))
    return {
        "num_prompts": len(prompts),
        "num_unique_exact": len(set(prompts)),
        "num_unique_normalized": len(set(normalized)),
        "num_unique_bow": len(set(key for key in bow_keys if key)),
        "quality_score": describe_numbers([item["quality_score"] for item in quality]),
        "complexity_score": describe_numbers([item["complexity_score"] for item in complexity]),
        "word_count": describe_numbers([item["word_count"] for item in quality]),
        "top_content_words": [
            {"token": token, "count": count}
            for token, count in vocab.most_common(30)
        ],
    }


def overlap_summary(prompts_a, prompts_b):
    exact_a = set(prompts_a)
    exact_b = set(prompts_b)
    norm_a = set(normalize_prompt(prompt) for prompt in prompts_a)
    norm_b = set(normalize_prompt(prompt) for prompt in prompts_b)
    bow_a = set(key for key in (canonical_bow_key(prompt) for prompt in prompts_a) if key)
    bow_b = set(key for key in (canonical_bow_key(prompt) for prompt in prompts_b) if key)
    vocab_a = set(token for prompt in prompts_a for token in content_tokens(prompt))
    vocab_b = set(token for prompt in prompts_b for token in content_tokens(prompt))

    exact_overlap = exact_a & exact_b
    norm_overlap = norm_a & norm_b
    bow_overlap = bow_a & bow_b

    return {
        "exact_overlap_count": len(exact_overlap),
        "normalized_overlap_count": len(norm_overlap),
        "bow_overlap_count": len(bow_overlap),
        "exact_overlap_ratio_a": len(exact_overlap) / max(1, len(exact_a)),
        "exact_overlap_ratio_b": len(exact_overlap) / max(1, len(exact_b)),
        "normalized_overlap_ratio_a": len(norm_overlap) / max(1, len(norm_a)),
        "normalized_overlap_ratio_b": len(norm_overlap) / max(1, len(norm_b)),
        "bow_overlap_ratio_a": len(bow_overlap) / max(1, len(bow_a)),
        "bow_overlap_ratio_b": len(bow_overlap) / max(1, len(bow_b)),
        "content_vocabulary_jaccard": jaccard(vocab_a, vocab_b),
        "shared_content_vocabulary_count": len(vocab_a & vocab_b),
        "unique_content_vocabulary_a": len(vocab_a),
        "unique_content_vocabulary_b": len(vocab_b),
        "sample_exact_overlap": sorted(exact_overlap)[:50],
        "sample_normalized_overlap": sorted(norm_overlap)[:50],
    }


def cross_similarity_report(prompts_a, prompts_b, pairs, top_k=50, embeddings_a=None, embeddings_b=None):
    cache_a = build_similarity_cache(prompts_a)
    cache_b = build_similarity_cache(prompts_b)
    structural_values = []
    token_values = []
    char_values = []
    bow_same_count = 0
    semantic_values = []
    top_structural = []
    top_semantic = []

    for i, j in pairs:
        token_sim = jaccard(cache_a["token_sets"][i], cache_b["token_sets"][j])
        char_sim = jaccard(cache_a["char_sets"][i], cache_b["char_sets"][j])
        bow_same = bool(cache_a["bow_keys"][i] and cache_a["bow_keys"][i] == cache_b["bow_keys"][j])
        structural = 0.55 * token_sim + 0.35 * char_sim + 0.10 * float(bow_same)
        structural_values.append(structural)
        token_values.append(token_sim)
        char_values.append(char_sim)
        bow_same_count += int(bow_same)

        row = {
            "i_a": i,
            "i_b": j,
            "structural_similarity": structural,
            "token_jaccard": token_sim,
            "char_ngram_jaccard": char_sim,
            "same_bow": bow_same,
            "prompt_a": prompts_a[i],
            "prompt_b": prompts_b[j],
        }
        update_top(top_structural, structural, row, top_k)

        if embeddings_a is not None and embeddings_b is not None:
            semantic = float(np.dot(embeddings_a[i], embeddings_b[j]))
            semantic_values.append(semantic)
            semantic_row = dict(row)
            semantic_row["semantic_similarity"] = semantic
            update_top(top_semantic, semantic, semantic_row, top_k)

    summary = {
        "sampled_cross_pair_count": len(pairs),
        "structural_similarity": describe_numbers(structural_values),
        "token_jaccard": describe_numbers(token_values),
        "char_ngram_jaccard": describe_numbers(char_values),
        "same_bow_pair_ratio": bow_same_count / max(1, len(pairs)),
        "cross_structural_homogeneity": float(np.mean(structural_values)) if structural_values else None,
        "cross_structural_distance": 1.0 - float(np.mean(structural_values)) if structural_values else None,
    }
    if embeddings_a is not None and embeddings_b is not None:
        semantic_mean = float(np.mean(semantic_values)) if semantic_values else None
        summary["semantic_similarity"] = describe_numbers(semantic_values)
        summary["cross_semantic_homogeneity"] = semantic_mean
        summary["cross_semantic_distance"] = 1.0 - semantic_mean if semantic_mean is not None else None

    return summary, sorted_heap_rows(top_structural), sorted_heap_rows(top_semantic)


def nearest_structural_matches(prompts_a, prompts_b, max_pairs_for_exact=5000000, top_k=50):
    if len(prompts_a) * len(prompts_b) > max_pairs_for_exact:
        return [], "skipped: too many cross pairs for exact structural nearest search"

    cache_a = build_similarity_cache(prompts_a)
    cache_b = build_similarity_cache(prompts_b)
    rows = []
    for i, prompt_a in enumerate(prompts_a):
        best = None
        for j, prompt_b in enumerate(prompts_b):
            token_sim = jaccard(cache_a["token_sets"][i], cache_b["token_sets"][j])
            char_sim = jaccard(cache_a["char_sets"][i], cache_b["char_sets"][j])
            bow_same = bool(cache_a["bow_keys"][i] and cache_a["bow_keys"][i] == cache_b["bow_keys"][j])
            score = 0.55 * token_sim + 0.35 * char_sim + 0.10 * float(bow_same)
            if best is None or score > best["structural_similarity"]:
                best = {
                    "i_a": i,
                    "i_b": j,
                    "structural_similarity": score,
                    "token_jaccard": token_sim,
                    "char_ngram_jaccard": char_sim,
                    "same_bow": bow_same,
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                }
        if best is not None:
            rows.append(best)

    rows.sort(key=lambda row: row["structural_similarity"], reverse=True)
    return rows[:top_k], None


def nearest_semantic_matches(prompts_a, prompts_b, embeddings_a, embeddings_b, top_k=50, block_size=512):
    heap = []
    for start in range(0, len(prompts_a), block_size):
        sims = embeddings_a[start:start + block_size] @ embeddings_b.T
        best_indices = sims.argmax(axis=1)
        best_scores = sims[np.arange(sims.shape[0]), best_indices]
        for offset, (j, score) in enumerate(zip(best_indices, best_scores)):
            i = start + offset
            row = {
                "i_a": i,
                "i_b": int(j),
                "semantic_similarity": float(score),
                "prompt_a": prompts_a[i],
                "prompt_b": prompts_b[int(j)],
            }
            update_top(heap, float(score), row, top_k)
    return sorted_heap_rows(heap)


def analyze_internal_if_requested(prompts, max_pairs, seed):
    pairs = sample_cross_pair_indices(len(prompts), len(prompts), max_pairs=max_pairs, seed=seed)
    pairs = [(i, j) for i, j in pairs if i < j]
    summary, _, _ = pairwise_report(prompts, pairs, top_k=0, semantic_embeddings=None)
    return summary


def compare_prompt_sets(args):
    prompts_a = load_prompts(
        args.prompts_a,
        split=args.split_a,
        prompt_column=args.prompt_column_a,
        max_prompts=args.max_prompts_a,
    )
    prompts_b = load_prompts(
        args.prompts_b,
        split=args.split_b,
        prompt_column=args.prompt_column_b,
        max_prompts=args.max_prompts_b,
    )
    if not prompts_a:
        raise ValueError("No prompts were loaded from prompts_a")
    if not prompts_b:
        raise ValueError("No prompts were loaded from prompts_b")

    semantic_embeddings_a = None
    semantic_embeddings_b = None
    semantic_error = None
    if args.semantic_model and args.semantic_model.lower() != "none":
        try:
            all_prompts = prompts_a + prompts_b
            embeddings = compute_text_embeddings(
                all_prompts,
                args.semantic_model,
                batch_size=args.semantic_batch_size,
                device=args.device,
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
            )
            semantic_embeddings_a = embeddings[:len(prompts_a)]
            semantic_embeddings_b = embeddings[len(prompts_a):]
        except Exception as exc:
            semantic_error = repr(exc)

    cross_pairs = sample_cross_pair_indices(
        len(prompts_a),
        len(prompts_b),
        max_pairs=args.max_cross_pairs,
        seed=args.seed,
    )
    cross_summary, top_structural, top_semantic = cross_similarity_report(
        prompts_a,
        prompts_b,
        cross_pairs,
        top_k=args.top_k,
        embeddings_a=semantic_embeddings_a,
        embeddings_b=semantic_embeddings_b,
    )

    nearest_struct_a_to_b, nearest_struct_error = nearest_structural_matches(
        prompts_a,
        prompts_b,
        max_pairs_for_exact=args.max_nearest_structural_pairs,
        top_k=args.top_k,
    )
    nearest_struct_b_to_a, nearest_struct_error_reverse = nearest_structural_matches(
        prompts_b,
        prompts_a,
        max_pairs_for_exact=args.max_nearest_structural_pairs,
        top_k=args.top_k,
    )
    nearest_sem_a_to_b = []
    nearest_sem_b_to_a = []
    if semantic_embeddings_a is not None and semantic_embeddings_b is not None:
        nearest_sem_a_to_b = nearest_semantic_matches(
            prompts_a,
            prompts_b,
            semantic_embeddings_a,
            semantic_embeddings_b,
            top_k=args.top_k,
            block_size=args.semantic_block_size,
        )
        nearest_sem_b_to_a = nearest_semantic_matches(
            prompts_b,
            prompts_a,
            semantic_embeddings_b,
            semantic_embeddings_a,
            top_k=args.top_k,
            block_size=args.semantic_block_size,
        )

    summary = {
        "source_a": str(args.prompts_a),
        "source_b": str(args.prompts_b),
        "dataset_a": prompt_set_basic_summary(prompts_a),
        "dataset_b": prompt_set_basic_summary(prompts_b),
        "overlap": overlap_summary(prompts_a, prompts_b),
        "cross_similarity": cross_summary,
        "internal_similarity_a": analyze_internal_if_requested(prompts_a, args.max_internal_pairs, args.seed)
        if args.include_internal else None,
        "internal_similarity_b": analyze_internal_if_requested(prompts_b, args.max_internal_pairs, args.seed + 1)
        if args.include_internal else None,
        "nearest_structural_error_a_to_b": nearest_struct_error,
        "nearest_structural_error_b_to_a": nearest_struct_error_reverse,
        "semantic_model": args.semantic_model,
        "semantic_error": semantic_error,
    }

    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(args.prompts_a, args.prompts_b)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "top_cross_structural_pairs.csv", top_structural)
    write_csv(out_dir / "nearest_structural_a_to_b.csv", nearest_struct_a_to_b)
    write_csv(out_dir / "nearest_structural_b_to_a.csv", nearest_struct_b_to_a)
    if semantic_embeddings_a is not None and semantic_embeddings_b is not None:
        write_csv(out_dir / "top_cross_semantic_pairs.csv", top_semantic)
        write_csv(out_dir / "nearest_semantic_a_to_b.csv", nearest_sem_a_to_b)
        write_csv(out_dir / "nearest_semantic_b_to_a.csv", nearest_sem_b_to_a)

    print(f"Loaded prompts A: {len(prompts_a)}")
    print(f"Loaded prompts B: {len(prompts_b)}")
    print(f"Report: {out_dir / 'summary.json'}")
    print(f"Top cross structural pairs: {out_dir / 'top_cross_structural_pairs.csv'}")
    if semantic_embeddings_a is not None and semantic_embeddings_b is not None:
        print(f"Top cross semantic pairs: {out_dir / 'top_cross_semantic_pairs.csv'}")
    elif semantic_error:
        print(f"Semantic similarity skipped: {semantic_error}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare similarity and overlap between two prompt sets.")
    parser.add_argument("--prompts_a", required=True, help="First json/jsonl/txt/csv prompt file or load_from_disk dataset")
    parser.add_argument("--prompts_b", required=True, help="Second json/jsonl/txt/csv prompt file or load_from_disk dataset")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--split_a", default="train")
    parser.add_argument("--split_b", default="train")
    parser.add_argument("--prompt_column_a", default=None)
    parser.add_argument("--prompt_column_b", default=None)
    parser.add_argument("--max_prompts_a", type=int, default=-1)
    parser.add_argument("--max_prompts_b", type=int, default=-1)
    parser.add_argument("--max_cross_pairs", type=int, default=300000)
    parser.add_argument("--max_internal_pairs", type=int, default=100000)
    parser.add_argument("--include_internal", type=int, default=1)
    parser.add_argument("--max_nearest_structural_pairs", type=int, default=5000000)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--semantic_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic_batch_size", type=int, default=64)
    parser.add_argument("--semantic_block_size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", type=int, default=0)
    parser.add_argument("--trust_remote_code", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    compare_prompt_sets(parse_args())
