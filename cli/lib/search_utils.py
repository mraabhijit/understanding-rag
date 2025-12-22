import json
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from cli.lib.gemini import client

DEFAULT_SEARCH_LIMIT = 3
SCORE_PRECISION = 3
DEFAULT_DESCRIPTION_LIMT = 100
MAX_CHUNK_SIZE = 200
OVERLAP_SIZE = 10
MAX_SEMANTIC_CHUNK_SIZE = 4
SEMANTIC_OVERLAP = 0

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def format_similarity_result(
    document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        document: Document dictionary
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": document['id'],
        "title": document['title'],
        "description": document['description'][:DEFAULT_DESCRIPTION_LIMT],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError('Input vectors must of same size.')
    
    if len(vec1) == 0:
        return []

    added_vectors = [0] * len(vec1)
    for i in range(len(vec1)):
        if not isinstance(vec1[i], (int, float)) and not isinstance(vec2[i], (int, float)):
            raise TypeError('Input vector elements must be of type int/float.')
        added_vectors[i] = vec1[i] + vec2[i]
    
    return added_vectors


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError('Input vectors must of same size.')
    
    if len(vec1) == 0:
        return []

    subtracted_vectors = [0] * len(vec1)
    for i in range(len(vec1)):
        if not isinstance(vec1[i], (int, float)) and not isinstance(vec2[i], (int, float)):
            raise TypeError('Input vector elements must be of type int/float.')
        subtracted_vectors[i] = vec1[i] - vec2[i]
    
    return subtracted_vectors

def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError('Input vectors must of same size.')
    
    if len(vec1) == 0:
        return 0.0

    product_vectors = [0] * len(vec1)
    for i in range(len(vec1)):
        if not isinstance(vec1[i], (int, float)) and not isinstance(vec2[i], (int, float)):
            raise TypeError('Input vector elements must be of type int/float.')
        product_vectors[i] = vec1[i] * vec2[i]
    
    return sum(product_vectors)

def eucleadian_norm(vec: list[float]) -> float:
    total = 0
    for x in vec:
        total += x ** 2

    return total ** 0.5

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError('Input vectors must of same size.')
    
    dot_product = dot(vec1, vec2)
    try: 
        return dot_product / (eucleadian_norm(vec1) * eucleadian_norm(vec2))
    except ZeroDivisionError:
        return 0.0

def enhance_query(query: str, enhancement: str) -> str:
    if enhancement == 'spell':
        contents = config.SPELL_CHECKER_PROMPT.format(query=query)
    elif enhancement == 'rewrite':
        contents = config.REWRITER_PROMPT.format(query=query)
    elif enhancement == 'expand':
        contents = config.EXPAND_PROMPT.format(query=query)

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=contents
    )
    return response.text
