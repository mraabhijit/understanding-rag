#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    DEFAULT_SEARCH_LIMIT,
    MAX_CHUNK_SIZE,
    chunk_command,
    OVERLAP_SIZE,
    MAX_SEMANTIC_CHUNK_SIZE,
    SEMANTIC_OVERLAP,
    semantic_command,
    embed_chunks_command,
    search_chunked,
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

    chunking_parser = subparsers.add_parser('chunk', help='Separate text into fixed sized chunks.')
    chunking_parser.add_argument(
        'text', type=str, help='Text to chunk',
    )
    chunking_parser.add_argument(
        '-c', '--chunk-size', dest="chunk_size", type=int, help='Max size of chunks', default=MAX_CHUNK_SIZE 
    )
    chunking_parser.add_argument(
        '-o', '--overlap', type=int, help='Number of words to overlap between chunks', default=OVERLAP_SIZE
    )

    semantic_chunk_parser = subparsers.add_parser('semantic_chunk', help='Separate text into fixed sized chunks.')
    semantic_chunk_parser.add_argument(
        'text', type=str, help='Text to chunk',
    )
    semantic_chunk_parser.add_argument(
        '-c', '--max-chunk-size', dest="max_chunk_size", type=int, help='Max size of chunks', default=MAX_SEMANTIC_CHUNK_SIZE
    )
    semantic_chunk_parser.add_argument(
        '-o', '--overlap', type=int, help='Number of words to overlap between chunks', default=SEMANTIC_OVERLAP
    )

    subparsers.add_parser('embed_chunks', help='Creates embeddings for the chunks from movies.')

    search_chunked_parser = subparsers.add_parser('search_chunked', help='Search relevant matches from chunked embeddings.')
    search_chunked_parser.add_argument(
        'query', type=str, help='Query to find matches for',
    )
    search_chunked_parser.add_argument(
        '--limit', '-l', type=int, help='Number of matches to retrieve.', default=5,
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
        case 'chunk':
            results = chunk_command(args.text, args.chunk_size, args.overlap)
            print(f'Chunking {len(args.text)} characters')
            for i, text in enumerate(results):
                print(f'{i+1}. {text}')
        case 'semantic_chunk':
            results = semantic_command(args.text, args.max_chunk_size, args.overlap)
            print(f'Semantically chunking {len(args.text)} characters')
            for i, text in enumerate(results):
                print(f'{i+1}. {text}')
        case 'embed_chunks':
            embed_chunks_command()
        case 'search_chunked':
            results = search_chunked(args.query, args.limit)
            for i, info in enumerate(results):
                print(f"\n{i+1}. {info['title']} (score: {info['score']:.4f})")
                print(f"   {info['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()