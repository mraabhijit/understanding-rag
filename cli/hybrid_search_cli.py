#!/usr/bin/env python3

import argparse

from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)

def main():
    parser = argparse.ArgumentParser(description='Hybrid Search CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    normalizer = subparsers.add_parser('normalize', help='Normalize scores')
    normalizer.add_argument(
        'scores', type=float, nargs='+', help='list of scores to normalize'
    )

    weighted_parser = subparsers.add_parser('weighted_search', help='Weighted search')
    weighted_parser.add_argument(
        'query', type=str, help='query to search'
    )
    weighted_parser.add_argument(
        '--alpha', type=float, help='alpha value for weighted search', default=0.5
    )
    weighted_parser.add_argument(
        '--limit', type=int, help='limit for weighted search', default=5
    )

    rrf_parser = subparsers.add_parser('rrf-search', help='RRF search')
    rrf_parser.add_argument(
        'query', type=str, help='query to search'
    )
    rrf_parser.add_argument(
        '--k', type=int, help='k value for rrf search', default=60
    )
    rrf_parser.add_argument(
        '--limit', type=int, help='limit for rrf search', default=5
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case 'normalize':
            normalized_scores = normalize_command(args.scores)
            if not normalized_scores:
                return
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case 'weighted_search':
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, score_info in enumerate(results):
                print(f"{i+1}. {score_info['title']}")
                print(f"   Hybrid Score: {score_info['hybrid_score']:.4f}")
                print(f"   BM25: {score_info['keyword_score']:.4f}, Semantic: {score_info['semantic_score']:.4f}")
                print(f"   {score_info['document'][:100]}...") 
        case 'rrf-search':
            results = rrf_search_command(args.query, args.k, args.limit, args.enhance)
            for i, score_info in enumerate(results):
                print(f"{i+1}. {score_info['title']}")
                print(f"   RRF Score: {score_info['rrf_score']:.4f}")
                print(f"   BM25 Rank: {score_info['bm25_rank']}, Semantic Rank: {score_info['semantic_rank']}")
                print(f"   {score_info['document'][:100]}...") 
        case _:
            parser.print_help()


if __name__ == '__main__':
    main()
