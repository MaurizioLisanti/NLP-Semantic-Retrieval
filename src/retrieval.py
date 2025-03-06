"""
retrieval.py
------------
Modulo per il retrieval di informazioni con FAISS, BM25 e TF-IDF.

Funzionalità principali:
✅ FAISS → Recupero basato su similarità semantica
✅ BM25 → Ricerca basata su parole chiave
✅ TF-IDF → Raffinamento della ricerca
✅ Fusione intelligente dei risultati

Autore: [Maurizio]

"""

import faiss
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import config  # Importiamo la configurazione globale

# ** Caricamento dei modelli e degli indici salvati**
def load_retrieval_models():
    """Carica FAISS, BM25 e TF-IDF dagli indici pre-addestrati."""
    print(" Caricamento degli indici FAISS, BM25 e TF-IDF...")

    # **Carica FAISS**
    index = faiss.read_index(config.DATA_PATH + "preprocessed/faiss_index.bin")
    embeddings = np.load(config.DATA_PATH + "preprocessed/embedding_vectors.npy")

    # **Carica BM25**
    with open(config.DATA_PATH + "preprocessed/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    # **Carica TF-IDF**
    with open(config.DATA_PATH + "preprocessed/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    print(" Indici caricati con successo!")
    return index, embeddings, bm25, tfidf_matrix

# **Funzione di retrieval con FAISS**
def faiss_search(query, model, index, top_k=10):
    """
    Effettua una ricerca con FAISS basata sugli embedding.

    Args:
        query (str): La query dell'utente.
        model (SentenceTransformer): Il modello NLP per generare gli embedding.
        index (faiss.IndexFlatL2): L'indice FAISS.
        top_k (int): Numero di risultati da restituire.

    Returns:
        list: Piatti più simili alla query.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    _, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return indices[0].tolist()

# ** Funzione di retrieval con BM25**
def bm25_search(query, bm25, piatti, top_k=10):
    """
    Effettua una ricerca con BM25 basata su parole chiave.

    Args:
        query (str): La query dell'utente.
        bm25 (BM25Okapi): Il modello BM25.
        piatti (list): Lista dei piatti disponibili.
        top_k (int): Numero di risultati da restituire.

    Returns:
        list: Piatti più pertinenti alla query.
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [piatti[i] for i in top_indices]

# ** Funzione di retrieval con TF-IDF**
def tfidf_search(query, tfidf_vectorizer, tfidf_matrix, piatti, top_k=10):
    """
    Effettua una ricerca con TF-IDF per migliorare la pertinenza.

    Args:
        query (str): La query dell'utente.
        tfidf_vectorizer (TfidfVectorizer): Il modello TF-IDF.
        tfidf_matrix (sparse matrix): Matrice TF-IDF addestrata.
        piatti (list): Lista dei piatti disponibili.
        top_k (int): Numero di risultati da restituire.

    Returns:
        list: Piatti più pertinenti alla query.
    """
    query_tfidf = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix * query_tfidf.T).toarray().flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [piatti[i] for i in top_indices]

# **Fusione dei risultati da FAISS, BM25 e TF-IDF**
def hybrid_search(query, model, index, bm25, tfidf_vectorizer, tfidf_matrix, piatti, top_k=10):
    """
    Combina i risultati di FAISS, BM25 e TF-IDF per ottenere il miglior retrieval.

    Args:
        query (str): La query dell'utente.
        model (SentenceTransformer): Il modello NLP per embedding.
        index (faiss.IndexFlatL2): L'indice FAISS.
        bm25 (BM25Okapi): Il modello BM25.
        tfidf_vectorizer (TfidfVectorizer): Il modello TF-IDF.
        tfidf_matrix (sparse matrix): Matrice TF-IDF.
        piatti (list): Lista dei piatti disponibili.
        top_k (int): Numero di risultati da restituire.

    Returns:
        list: Piatti più pertinenti alla query.
    """
    faiss_results = faiss_search(query, model, index, top_k)
    bm25_results = bm25_search(query, bm25, piatti, top_k)
    tfidf_results = tfidf_search(query, tfidf_vectorizer, tfidf_matrix, piatti, top_k)

    # **Fusion dei risultati** (prendiamo quelli più frequenti tra i tre metodi)
    combined_results = list(set(faiss_results + bm25_results + tfidf_results))

    return combined_results[:top_k]

# ** Esecuzione del retrieval**
if __name__ == "__main__":
    print(" Avvio del processo di retrieval...")

    # Carica i dati e i modelli
    df_piatti = pd.read_csv(config.DATA_PATH + "preprocessed/piatti_cleaned.csv")
    piatti = df_piatti["piatto"].tolist()

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index, embeddings, bm25, tfidf_matrix = load_retrieval_models()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(piatti)

    # Esempio di query
    query = "Vorrei un piatto a base di pesce e spezie"
    results = hybrid_search(query, model, index, bm25, tfidf_vectorizer, tfidf_matrix, piatti)

    print(f" Risultati per la query '{query}': {results}")
