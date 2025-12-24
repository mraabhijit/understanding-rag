import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config

from lib.hybrid_search import HybridSearch
from lib.search_utils import (
    load_movies,
)
from lib.gemini import client


def rag_command(query: str):
    documents = load_movies()
    hybrid = HybridSearch(documents)
    docs = hybrid.rrf_search(
        query=query,
        limit=5
    )
    print("Search Results:")
    for doc in docs:
        print(f"  - {doc['title']}")

    contents = config.RAG_PROMPT.format(
        query=query,
        docs=docs,
    )

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=contents,
    )
    print("\nRAG Response:")
    print(response.text)
