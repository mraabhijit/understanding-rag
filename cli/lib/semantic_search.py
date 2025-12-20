import os

import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import (
    CACHE_DIR,
    load_movies,
    format_similarity_result,
    DEFAULT_SEARCH_LIMIT,
)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, 'movie_embeddings.npy')

    def generate_embedding(self, text: str) -> list:
        text = text.strip()
        if text == '':
            raise ValueError('Input string empty.')
        
        embeddings = self.model.encode([text])
        return embeddings[0]


    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_strings = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_str = f"{doc['title']}: {doc['description']}"
            doc_strings.append(doc_str)

        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        with open(self.embeddings_path, 'wb') as f:
            np.save(f, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if os.path.exists(self.embeddings_path) and os.path.getsize(self.embeddings_path) > 0:
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        score_tuples = []
        for i in range(len(self.documents)):
            similarity_score = cosine_similarity(self.embeddings[i], query_embedding)
            score_tuples.append((similarity_score, self.documents[i]))

        sorted_scores = sorted(score_tuples, reverse=True, key=lambda x: x[0])
        results = []
        for score, doc in sorted_scores[:limit]:
            formatted_result = format_similarity_result(
                document=doc,
                score=score,
            )
            results.append(formatted_result)

        return results


def verify_model():
    search = SemanticSearch()
    print(f'Model loaded: {search.model}')
    print(f'Max sequence length: {search.model.max_seq_length}')


def embed_text(text: str):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    sem = SemanticSearch()
    documents = load_movies()
    embeddings = sem.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    sem = SemanticSearch()
    documents = load_movies()
    _ = sem.load_or_create_embeddings(documents)
    results = sem.search(query, limit)
    return results
