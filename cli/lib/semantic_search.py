import os
import re
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import (
    CACHE_DIR,
    load_movies,
    format_similarity_result,
    DEFAULT_SEARCH_LIMIT,
    MAX_CHUNK_SIZE,
    OVERLAP_SIZE,
    MAX_SEMANTIC_CHUNK_SIZE,
    SEMANTIC_OVERLAP,
    SCORE_PRECISION,
)

class SemanticSearch:
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, 'chunk_embeddings.npy')
        self.chunk_metadata_path = os.path.join(CACHE_DIR, 'chunk_metadata.json')

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = []
        metadata = []
        for i, doc in enumerate(documents):
            self.document_map[doc['id']] = doc
            if len(doc['description']) == 0:
                continue

            semantic_chunks = semantic_command(doc['description'], 4, 1)
            total_chunks = len(semantic_chunks)
            for j, chunk in enumerate(semantic_chunks):
                chunks.append(chunk)
                metadata.append({
                    'movie_idx': i,
                    'chunk_idx': j,
                    'total_chunks': total_chunks
                })

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        with open(self.chunk_embeddings_path, 'wb') as f:
            np.save(f, self.chunk_embeddings)

        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if (
            os.path.exists(self.chunk_embeddings_path)
            and os.path.exists(self.chunk_metadata_path)
            and os.path.getsize(self.chunk_embeddings_path)
            and os.path.getsize(self.chunk_metadata_path)
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                self.chunk_metadata = metadata_dict['chunks']
            
            return self.chunk_embeddings
    
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        # For each chunk embedding, calculate similarity and collect info
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(chunk_embedding, query_embedding)
            meta = self.chunk_metadata[i]
            chunk_scores.append({
                'chunk_idx': meta['chunk_idx'],
                'movie_idx': meta['movie_idx'],
                'score': similarity
            })

        # Aggregate: for each movie_idx, keep the highest scoring chunk
        movie_score_map = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            score = chunk_score['score']
            if (movie_idx not in movie_score_map) or (score > movie_score_map[movie_idx]['score']):
                movie_score_map[movie_idx] = {
                    'score': score,
                    'chunk_idx': chunk_score['chunk_idx']
                }

        # Sort by score descending, take top limit
        sorted_movies = sorted(movie_score_map.items(), key=lambda x: x[1]['score'], reverse=True)[:limit]
        results = []
        for movie_idx, info in sorted_movies:
            doc = self.documents[movie_idx]
            # Find the chunk_metadata entry for this movie's highest scoring chunk
            chunk_meta = None
            for meta in self.chunk_metadata:
                if meta['movie_idx'] == movie_idx and meta['chunk_idx'] == info['chunk_idx']:
                    chunk_meta = meta
                    break
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'document': doc['description'],
                'score': round(info['score'], SCORE_PRECISION),
                'metadata': chunk_meta if chunk_meta is not None else {}
            })
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


def chunk_command(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> list[str]:
    words = text.split()
    total_words = len(words)
    results = []
    i = 0
    while i < total_words:
        chunked_text = ' '.join(words[i : i + chunk_size])
        results.append(chunked_text)
        i += (chunk_size - overlap)

        if i >= total_words or chunk_size <= overlap:
            break

    return results


def semantic_command(text: str, chunk_size: int = MAX_SEMANTIC_CHUNK_SIZE, overlap: int = SEMANTIC_OVERLAP) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError('Chunk Size must be greater than overlap.')
    
    # 1. Strip leading and trailing whitespace from the input text before using the regex to split sentences.
    text = text.strip()
    
    # 2. If there's nothing left after stripping, return an empty list.
    if not text:
        return []
    
    # 3. After splitting sentences...
    sentences = re.split(pattern=r"(?<=[.!?])\s+", string=text)
    
    # 4. Strip leading and trailing whitespace from each sentence (Requirement 4 check)
    # And filter out empty ones
    sentences = [s.strip() for s in sentences if s.strip()]

    # 3. ...if there's only one sentence and it doesn't end with a punctuation mark like ., !, or ?, treat the whole text as one sentence.
    if len(sentences) == 1:
        last_char = sentences[0][-1]
        if last_char not in ('.', '!', '?'):
            sentences = [text]

    total_sentences = len(sentences)
    results = []
    i = 0
    while i < total_sentences:
        chunked_text = ' '.join(sentences[i : i + chunk_size])
        
        # Strip leading and trailing whitespace from the chunk (Requirement 4/5)
        cleaned_text_chunk = chunked_text.strip()
        
        # 5. Only use chunks that still have content after the stripping.
        if cleaned_text_chunk:
            results.append(cleaned_text_chunk)
            
        i += (chunk_size - overlap)
        if i >= total_sentences:
            break

    return results


def embed_chunks_command():
    c_sem = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = c_sem.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int = 5):
    c_sem = ChunkedSemanticSearch()
    documents = load_movies()
    _ = c_sem.load_or_create_chunk_embeddings(documents)
    results = c_sem.search_chunks(query, limit)
    return results
