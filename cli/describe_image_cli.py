#!/usr/bin/env python

import os
import sys

import argparse
import mimetypes
from google.genai import types

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import config
from lib.gemini import client


PROMPT = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary

rewritten query:
"""

def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")

    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    
    contents = ''
    with open(args.image, "rb") as f:
        contents = f.read()

    parts = [
        PROMPT,
        types.Part.from_bytes(data=contents, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=parts,
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == '__main__':
    main()
