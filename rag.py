import json, os
from typing import List, Dict, Any
import numpy as np

class KnowledgeRAG:
    def __init__(self, path: str, embedding_model: str):
        self.path = path
        with open(path, 'r', encoding='utf-8') as f:
            self.docs = json.load(f)
        self.encoder = None
        self.index = None
        self._embedding_model_name = embedding_model

    def _ensure_index(self):
        if self.index is not None:
            return
        from sentence_transformers import SentenceTransformer
        import faiss
        self.encoder = SentenceTransformer(self._embedding_model_name)
        texts = [self._text(d) for d in self.docs]
        emb = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype='float32')
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    @staticmethod
    def _text(d: Dict[str, Any]) -> str:
        return ' | '.join([
            d.get('crop',''), d.get('disease',''), d.get('type',''),
            d.get('symptoms',''), d.get('management',''), d.get('prevention','')
        ])

    def retrieve(self, crop: str, disease: str | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_index()
        q = f"crop: {crop}; disease: {disease or 'unknown'}"
        qemb = self.encoder.encode([q], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(qemb, dtype='float32'), min(top_k, len(self.docs)))
        out=[]
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                out.append({'score': float(score), 'document': self.docs[int(idx)]})
        return out

def severity_stage(severity: float) -> str:
    if severity < 10: return 'early'
    if severity < 30: return 'moderate'
    return 'advanced'

def build_prognosis(severity: float, environment: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    # Decision-support projection only. It is intentionally not presented as a validated biological forecast.
    humidity = (environment or {}).get('humidity_pct')
    multiplier = 1.15 if isinstance(humidity, (int,float)) and humidity >= 75 else 1.07
    values=[]
    for days in (3,7,14):
        projected = min(100.0, severity * (multiplier ** (days/3)))
        values.append({'days': days, 'severity_percent': round(projected,1), 'basis':'heuristic scenario; replace with longitudinally trained model'})
    return values
