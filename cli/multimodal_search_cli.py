#!/usr/bin/env python

import argparse

from lib.multimodal_search import (
    verify_image_embedding,
    image_search_command,
)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        'verify_image_embedding',
        help="Verify image embeddings for passed image path",
    )
    verify_image_embedding_parser.add_argument(
        'image_path', type=str, help="Path to image to embed and verify"
    )

    image_search_parser = subparsers.add_parser(
        'image_search',
        help="Retrieve movies relevant to image",
    )
    image_search_parser.add_argument(
        'image_path', type=str, help="Path to image to search the movies relevant."
    )


    args = parser.parse_args()

    match args.command:
        case 'verify_image_embedding':
            verify_image_embedding(args.image_path)
        case 'image_search':
            results = image_search_command(args.image_path)
            for i, doc in enumerate(results):
                print(f"{i+1}. {doc['title']} (similarity: {doc['similarity_score']:.3f})")
                print(f"   {doc['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == '__main__':
    main()
