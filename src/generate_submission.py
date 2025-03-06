"""
generate_submission.py
----------------------
Pipeline completa per generare la submission per la competizione Kaggle.

Funzionalità principali:
✅ Carica i dati e i modelli FAISS, BM25 e TF-IDF
✅ Effettua il retrieval con FAISS + BM25 + TF-IDF
✅ Applica il reranking con LLM per migliorare la qualità delle risposte
✅ Genera il file `submission.csv` pronto per essere caricato su Kaggle

Autore: [Maurizio]

"""

import pandas as pd
import config
from retrieval import hybrid_search
from reranking import rerank_results, load_reranker
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# **Caricare i dati della competizione**
def load_data():
    """
    Carica il database dei piatti, il mapping degli ID e il file delle domande.

    Returns:
        df_piatti (DataFrame): Dataset dei piatti con ID.
        dish_mapping (dict): Dizionario {piatto: ID}.
        domande_data (list): Lista di domande.
    """
    print("📥 Caricamento dei dati della competizione...")

    df_piatti = pd.read_csv(config.DATA_PATH + "preprocessed/piatti_cleaned.csv")

    with open(config.DATA_PATH + "dish_mapping.json", "r", encoding="utf-8") as f:
        dish_mapping = json.load(f)

    with open(config.DATA_PATH + "domande.json", "r", encoding="utf-8") as f:
        domande_data = json.load(f)

    print(f" {len(df_piatti)} piatti caricati con successo!")
    print(f" {len(domande_data)} domande caricate con successo!")

    return df_piatti, dish_mapping, domande_data

# ** Caricare i modelli di retrieval**
def load_retrieval_models():
    """
    Carica i modelli FAISS, BM25 e TF-IDF dagli indici pre-addestrati.

    Returns:
        index (FAISS): Indice FAISS per il retrieval veloce.
        bm25 (BM25Okapi): Modello BM25 per il retrieval testuale.
        tfidf_vectorizer (TfidfVectorizer): Vettorizzatore TF-IDF.
        tfidf_matrix (sparse matrix): Matrice TF-IDF.
    """
    print(" Caricamento degli indici FAISS, BM25 e TF-IDF...")

    # **Carica FAISS**
    index = faiss.read_index(config.DATA_PATH + "preprocessed/faiss_index.bin")

    # **Carica BM25**
    with open(config.DATA_PATH + "preprocessed/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    # **Carica TF-IDF**
    with open(config.DATA_PATH + "preprocessed/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(pd.read_csv(config.DATA_PATH + "preprocessed/piatti_cleaned.csv")["piatto"].tolist())

    print(" Indici caricati con successo!")
    return index, bm25, tfidf_vectorizer, tfidf_matrix

# **Generare la submission**
def generate_submission():
    """
    Esegue la pipeline completa di retrieval + reranking e genera il file `submission.csv`.
    """
    print(" Avvio del processo di generazione della submission...")

    # **Caricare i dati**
    df_piatti, dish_mapping, domande_data = load_data()
    piatti = df_piatti["piatto"].tolist()

    # **Caricare i modelli di retrieval**
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index, bm25, tfidf_vectorizer, tfidf_matrix = load_retrieval_models()

    # **Caricare il modello di reranking**
    reranker = load_reranker()

    results = []

    for entry in domande_data:
        domanda = entry["domanda"]
        row_id = entry["row_id"]

        # **Step 1: Retrieval con FAISS + BM25 + TF-IDF**
        retrieved_results = hybrid_search(
            domanda, model, index, bm25, tfidf_vectorizer, tfidf_matrix, piatti, top_k=config.TOP_K_RETRIEVAL
        )

        # **Step 2: Reranking con LLM**
        final_results = rerank_results(domanda, retrieved_results, reranker, top_k=config.TOP_K_RERANKING)

        # **Step 3: Convertire i piatti in ID**
        matched_ids = [dish_mapping[p] for p in final_results if p in dish_mapping]

        results.append({"row_id": row_id, "result": ",".join(map(str, matched_ids))})

    # **Step 4: Creare il file CSV**
    submission_df = pd.DataFrame(results)
    submission_path = f"{config.SUBMISSION_PATH}/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f" Submission generata con successo! File salvato in: {submission_path}")

# ** Eseguire lo script**
if __name__ == "__main__":
    generate_submission()
