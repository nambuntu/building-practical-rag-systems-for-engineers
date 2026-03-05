from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import benchmark_bruteforce, benchmark_faiss
from dataset import Dataset, build_dataset, load_dataset, save_dataset
from report import render_report

DEFAULT_DATASET_PATH = Path("data/ch04_dataset.npz")


def _dataset_matches(dataset: Dataset, *, n: int, d: int, q: int) -> bool:
    return (
        dataset.vectors.shape == (n, d)
        and dataset.queries.shape == (q, d)
        and len(dataset.doc_ids) == n
        and len(dataset.relevant_doc_ids) == q
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 4 similarity search benchmark lab.")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--d", type=int, default=384)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--q", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", choices=["flat", "ivf"], default="ivf")
    parser.add_argument("--nlist", type=int, default=200)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--reuse-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--brute-mode", choices=["vectorized", "naive"], default="vectorized")
    args = parser.parse_args(argv)

    dataset_path = DEFAULT_DATASET_PATH
    dataset: Dataset
    loaded_from_cache = False

    if args.reuse_dataset and dataset_path.exists():
        candidate = load_dataset(dataset_path)
        if _dataset_matches(candidate, n=args.n, d=args.d, q=args.q):
            dataset = candidate
            loaded_from_cache = True
        else:
            dataset = build_dataset(
                n=args.n,
                d=args.d,
                clusters=args.clusters,
                q=args.q,
                noise=args.noise,
                seed=args.seed,
            )
            save_dataset(dataset_path, dataset)
    else:
        dataset = build_dataset(
            n=args.n,
            d=args.d,
            clusters=args.clusters,
            q=args.q,
            noise=args.noise,
            seed=args.seed,
        )
        save_dataset(dataset_path, dataset)

    print(f"Dataset: {'loaded from cache' if loaded_from_cache else 'generated'} ({dataset_path})")

    brute_result = benchmark_bruteforce(
        vectors=dataset.vectors,
        doc_ids=dataset.doc_ids,
        queries=dataset.queries,
        relevant_doc_ids=dataset.relevant_doc_ids,
        k=args.k,
        mode=args.brute_mode,
    )

    faiss_result = benchmark_faiss(
        vectors=dataset.vectors,
        doc_ids=dataset.doc_ids,
        queries=dataset.queries,
        relevant_doc_ids=dataset.relevant_doc_ids,
        k=args.k,
        index_type=args.index,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )

    config = {
        "n": args.n,
        "d": args.d,
        "clusters": args.clusters,
        "q": args.q,
        "k": args.k,
        "noise": args.noise,
        "seed": args.seed,
        "index": args.index,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
    }
    print()
    print(render_report(config=config, brute_result=brute_result, faiss_result=faiss_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
