from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import (
    cosine_similarity,
    load_movies,
)


class MultimodalSearch:
    def __init__(self, model_name: str = 'clip-ViT-B-32', docs: list = None):
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = None
        self.text_embeddings = None
        if docs:
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
            self.text_embeddings = self.model.encode(
                self.texts, show_progress_bar=True
            )

    def embed_image(self, img_path: str):
        img = Image.open(img_path)
        embeddings = self.model.encode([img])
        return embeddings[0]
    
    def search_with_image(self, img_path: str, limit: int = 5):
        img_embedding = self.embed_image(img_path)
        cosine_scores = []
        if not self.text_embeddings:
            raise RuntimeError('Initialize Multimodal search with list of documents to create text embeddings')

        for i, text_embedding in enumerate(self.text_embeddings):
            cosine = cosine_similarity(text_embedding.tolist(), img_embedding.tolist())
            cosine_scores.append(
                {
                    'doc_id': self.docs[i]['id'],
                    'title': self.docs[i]['title'],
                    'description': self.docs[i]['description'],
                    'similarity_score': cosine
                }
            )

        # sort scores
        return sorted(cosine_scores, key=lambda x: x['similarity_score'], reverse=True)[:limit]


def verify_image_embedding(img_path: str):
    ms = MultimodalSearch()
    embedding = ms.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(img_path: str):
    documents = load_movies()
    ms = MultimodalSearch(docs=documents)
    results = ms.search_with_image(img_path)

    return results
