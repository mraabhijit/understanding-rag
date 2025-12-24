import os
import json

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import (
    load_movies,
    enhance_query,
    rerank_individual,
    rerank_batch,
    llm_evaluate,
)

from sentence_transformers import CrossEncoder


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_scores = self._bm25_search(query, 500 * limit)
        semantic_scores = self.semantic_search.search_chunks(query, 500 * limit)
        normalized_bm25_scores = normalize_command([score_info['score'] for score_info in bm25_scores])
        normalized_semantic_scores = normalize_command([score_info['score'] for score_info in semantic_scores])

        combined_scores = {}
        for i, score in enumerate(normalized_bm25_scores):
            score_data = bm25_scores[i]
            combined_scores[score_data['id']] = {
                'id': score_data['id'],
                'title': score_data['title'],
                'document': score_data['document'],
                'keyword_score': score,
                'semantic_score': 0,
            }

        for i, score in enumerate(normalized_semantic_scores):
            score_data = semantic_scores[i]
            if score_data['id'] in combined_scores:
                combined_scores[score_data['id']]['semantic_score'] = score
            else:
                combined_scores[score_data['id']] = {
                    'id': score_data['id'],
                    'title': score_data['title'],
                    'document': score_data['document'],
                    'keyword_score': 0,
                    'semantic_score': score,
                }

        for scores in combined_scores.values():
            scores['hybrid_score'] = hybrid_score(scores['keyword_score'], scores['semantic_score'], alpha)
        
        return sorted(combined_scores.values(), key=lambda x: x['hybrid_score'], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 10):
        bm25_scores = self._bm25_search(query, 500 * limit)
        semantic_scores = self.semantic_search.search_chunks(query, 500 * limit)

        ranked_bm25_scores = sorted(bm25_scores, key=lambda x: x['score'], reverse=True)[:limit]
        ranked_semantic_scores = sorted(semantic_scores, key=lambda x: x['score'], reverse=True)[:limit]

        rrf_results = {}
        for i, data in enumerate(ranked_bm25_scores):
            rrf_results[data['id']] = {
                'id': data['id'],
                'title': data['title'],
                'document': data['document'],
                'keyword_rrf_score': rrf_score(i+1),
                'bm25_rank': i+1,
                'semantic_rank': 'NA',
                'semantic_rrf_score': 0,
            }

        for i, data in enumerate(ranked_semantic_scores):
            if data['id'] in rrf_results:
                rrf_results[data['id']]['semantic_rrf_score'] = rrf_score(i+1)
                rrf_results[data['id']]['semantic_rank'] = i+1
            else:
                rrf_results[data['id']] = {
                    'id': data['id'],
                    'title': data['title'],
                    'document': data['document'],
                    'keyword_rrf_score': 0,
                    'bm25_rank': 'NA',
                    'semantic_rrf_score': rrf_score(i+1),
                    'semantic_rank': i+1,
                }

        for scores in rrf_results.values():
            scores['rrf_score'] = scores['keyword_rrf_score'] + scores['semantic_rrf_score']

        return sorted(rrf_results.values(), key=lambda x: x['rrf_score'], reverse=True)[:limit]

def normalize_command(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]

def hybrid_score(bm25_score: float, semantic_score: float, alpha=0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5):
    documents = load_movies()
    hybrid = HybridSearch(documents)
    results = hybrid.weighted_search(query, alpha, limit)
    return results

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search_command(query: str, k: int = 60, limit: int = 10, enhance: str = None, rerank_method: str = None, evaluate: bool = False):
    print(f"Original query: {query}")
    documents = load_movies()
    hybrid = HybridSearch(documents)
    if enhance is not None:
        original_query = query
        query = enhance_query(query, enhance)
        print(f"Enhanced query ({enhance}): '{original_query}' -> '{query}'\n")
    results = hybrid.rrf_search(query, k, limit)

    if evaluate:
        llm_ranks = json.loads(llm_evaluate(query, results))
        for i, (rank, doc) in enumerate(zip(llm_ranks, results)):
            print(f"{i+1}. {doc['title']}: {rank}/3")

    if not rerank_method:
        return results

    if rerank_method == 'individual':
        return individual_reranker(query, results)
    elif rerank_method == 'batch':
        return batch_reranker(query, results)
    # Cross encoder reranker
    return cross_encoder_reranker(query, results)

def individual_reranker(query: str, doc_list: list[dict]) -> list[dict]:
    for i, doc in enumerate(doc_list):
        rerank_score = rerank_individual(query, doc)
        doc_list[i]['rerank_score'] = rerank_score

    return sorted(doc_list, key=lambda x: x['rerank_score'], reverse=True)


def batch_reranker(query: str, doc_list: list[dict]) -> list[dict]:
    rerank_response = rerank_batch(query, doc_list)
    if not rerank_response.strip():
        print("Error: rerank_batch returned empty response")
        return doc_list

    try:
        rerank_result = json.loads(rerank_response)
        # Initialize all doc_list with a default high rank (lower priority)
        for result in doc_list:
            result['rerank_rank'] = len(doc_list) + 1
        
        # Assign reranked positions to matching doc_list
        for rank, doc_id in enumerate(rerank_result):
            for result in doc_list:
                if doc_id == result['id']:
                    result['rerank_rank'] = rank + 1
                    break
        
        sorted_doc_list = sorted(doc_list, key=lambda x: x['rerank_rank'])
    except json.JSONDecodeError as e:
        print(f"Error parsing rerank response: {e}")
        print(f"Response was: {rerank_response}")
        sorted_doc_list = doc_list

    return sorted_doc_list


def cross_encoder_reranker(query: str, doc_list: list[dict]) -> list[dict]:
    pairs = [(query, f"{doc.get('title', '')} - {doc.get('document', '')}") for doc in doc_list]
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(doc_list, scores):
        doc['cross_encoder_score'] = score

    return sorted(doc_list, key=lambda x: x['cross_encoder_score'], reverse=True)

