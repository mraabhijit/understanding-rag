#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    DEFAULT_SEARCH_LIMIT,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )
    subparsers.add_parser('verify', help='Verify the embedding model.')

    embed_parser = subparsers.add_parser('embed_text', help='Create embeddings for input text.')
    embed_parser.add_argument(
        'text', type=str, help='Text string to create embeddings for.'
    )

    subparsers.add_parser('verify_embeddings', help='Verify the embeddings of movies.')

    query_embed_parser = subparsers.add_parser('embedquery', help='Embeds the query passed.')
    query_embed_parser.add_argument(
        'query', type=str, help='Query to get embeddings for.'
    )

    search_parser = subparsers.add_parser('search', help='Get relevant matches for query.')
    search_parser.add_argument(
        'query', type=str, help='Query to get matches for.'
    )
    search_parser.add_argument(
        '--limit', '-l', type=int, help='Number of matches to retrieve.', default=DEFAULT_SEARCH_LIMIT,
    )

    args = parser.parse_args()

    match args.command:
        case 'verify':
            verify_model()
        case 'embed_text':
            embed_text(args.text)
        case 'verify_embeddings':
            verify_embeddings()
        case 'embedquery':
            embed_query_text(args.query)
        case 'search':
            results = search_command(args.query, args.limit)
            for i, item in enumerate(results):
                print(f"{i+1}. {item['title']} (score: {item['score']:.4f})\n   {item['description']} ...\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()