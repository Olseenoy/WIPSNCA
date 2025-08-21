import os
import numpy as np
import pandas as pd
from typing import List, Dict
from joblib import Memory

cache = Memory(location='./.cache', verbose=0)

try:
    from sentence_transformers import SentenceTransformer
    S_MODEL = os.getenv('EMBED_MODEL', 'all-mpnet-base-v2')
    EMB = SentenceTransformer(S_MODEL)
    _USE_EMB = True
except Exception:
    # fallback to sklearn TF-IDF
    _USE_EMB = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    _VEC = None

@cache.cache
def load_dataset(path: str = 'data/sample_nc.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['detected_at'])
    df['doc'] = df['title'].fillna('') + "\n" + df['description'].fillna('')
    return df

class SimilarityEngine:
    def __init__(self, data_path: str = 'data/sample_nc.csv'):
        self.df = load_dataset(data_path)
        if _USE_EMB:
            texts = self.df['doc'].tolist()
            self.vectors = EMB.encode(texts, normalize_embeddings=True)
        else:
            global _VEC
            _VEC = TfidfVectorizer(max_features=2000)
            self.tfidf = _VEC.fit_transform(self.df['doc'].tolist())

    def topk(self, query: str, k: int = 5) -> List[Dict]:
        q = query
        if _USE_EMB:
            qv = EMB.encode([q], normalize_embeddings=True)
            sims = np.dot(self.vectors, qv[0])
            idx = np.argsort(-sims)[:k]
            results = []
            for i in idx:
                results.append({
                    'score': float(sims[i]),
                    'id': str(self.df.iloc[i].get('id', i)),
                    'title': self.df.iloc[i]['title'],
                    'description': self.df.iloc[i]['description'],
                    'site': self.df.iloc[i].get('site'),
                    'line': self.df.iloc[i].get('line'),
                    'defect_type': self.df.iloc[i].get('defect_type')
                })
            return results
        else:
            qv = _VEC.transform([q])
            from sklearn.metrics.pairwise import linear_kernel
            sims = linear_kernel(qv, self.tfidf).flatten()
            idx = sims.argsort()[::-1][:k]
            results = []
            for i in idx:
                results.append({
                    'score': float(sims[i]),
                    'id': str(self.df.iloc[i].get('id', i)),
                    'title': self.df.iloc[i]['title'],
                    'description': self.df.iloc[i]['description'],
                    'site': self.df.iloc[i].get('site'),
                    'line': self.df.iloc[i].get('line'),
                    'defect_type': self.df.iloc[i].get('defect_type')
                })
            return results