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


def summarize_command(query: str, limit: int):
    documents = load_movies()
    hybrid = HybridSearch(documents)
    results = hybrid.rrf_search(
        query=query,
        limit=limit,
    )
    print("Search Results:")
    for doc in results:
        print(f"  - {doc['title']}")

    contents = config.SUMMARIZE_PROMPT.format(
        query=query,
        results=results,
    )

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=contents,
    )
    print("\nLLM Summary:")
    print(response.text)


def citation_command(query: str, limit: int):
    documents = load_movies()
    hybrid = HybridSearch(documents)
    docs = hybrid.rrf_search(
        query=query,
        limit=limit,
    )
    print("Search Results:")
    for doc in docs:
        print(f"  - {doc['title']}")

    contents = config.CITATIONS_PROMPT.format(
        query=query,
        documents=docs,
    )

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=contents,
    )
    print("\nLLM Answer:")
    print(response.text)


def question_answer_command(query: str, limit: int):
    documents = load_movies()
    hybrid = HybridSearch(documents)
    docs = hybrid.rrf_search(
        query=query,
        limit=limit,
    )
    print("Search Results:")
    for doc in docs:
        print(f"  - {doc['title']}")

    contents = config.QUESTION_ANSWERING_PROMPT.format(
        question=query,
        context=docs,
    )

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=contents,
    )
    print("\nAnswer:")
    print(response.text)
