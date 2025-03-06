"""
preprocessing.py
----------------
Questo modulo si occupa della pulizia e della preparazione dei dati per la pipeline di retrieval e ranking.
Viene eseguito prima del processo di ricerca per garantire che i dati siano coerenti, puliti e ottimizzati.

Funzionalità principali:
✅ Caricamento e pulizia dei dati
✅ Rimozione di duplicati e dati mancanti
✅ Creazione e salvataggio di indici FAISS, BM25 e TF-IDF
✅ Salvataggio degli embedding precomputati

Autore: [Maurizio]

"""

import pandas as pd
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import config  # Importa i parametri di configurazione

# **Caricare i dati grezzi**
def load_data():
    """Carica i file della competizione e li restituisce come oggetti Python."""
    print(" Caricamento dei dati...")
    
    # Carica i file CSV e JSON
    df_piatti = pd.read_csv(config.DATA_PATH + "database_piatti_con_id.csv")

    with open(config.DATA_PATH + "dish_mapping.json", "r", encoding="utf-8") as f:
        dish_mapping = json.load(f)

    with open(config.DATA_PATH + "domande.json", "r", encoding="utf-8") as f:
        domande_data = json.load(f)

    print(" Dati caricati con successo!")
    return df_piatti, dish_mapping, domande_data

# **Pulizia e normalizzazione dei dati**
def clean_data(df_piatti):
    """Pulisce i dati rimuovendo valori nulli, normalizzando i testi e rimuovendo duplicati."""
    print("🧹 Pulizia e normalizzazione dei dati...")

    # Rimuove righe con piatti mancanti
    df_piatti.dropna(subset=["piatto"], inplace=True)

    # Converte tutto in minuscolo e rimuove spazi extra
    df_piatti["piatto"] = df_piatti["piatto"].str.lower().str.strip()

    # Rimuove eventuali duplicati basati sul nome del piatto
    df_piatti.drop_duplicates(subset=["piatto"], inplace=True)

    print(" Dati puliti con successo! Numero di piatti unici:", len(df_piatti))
    return df_piatti

# **Creazione e salvataggio dell’indice FAISS**
def create_faiss_index(df_piatti):
    """Crea e salva l'indice FAISS per il retrieval veloce basato su embedding."""
    print(" Creazione dell'indice FAISS...")

    # Carica il modello NLP per generare gli embedding
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Calcola gli embedding dei piatti
    piatti_embeddings = model.encode(df_piatti["piatto"].tolist(), convert_to_numpy=True)

    # Creazione dell’indice FAISS
    d = piatti_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(piatti_embeddings)

    # Salvataggio dell'indice FAISS su file
    faiss.write_index(index, config.DATA_PATH + "preprocessed/faiss_index.bin")
    np.save(config.DATA_PATH + "preprocessed/embedding_vectors.npy", piatti_embeddings)

    print(" Indice FAISS salvato con successo!")

# **Creazione e salvataggio di BM25**
def create_bm25(df_piatti):
    """Crea e salva il modello BM25 per il retrieval testuale."""
    print(" Creazione dell'indice BM25...")

    # Tokenizzazione dei piatti per BM25
    tokenized_piatti = [p.split() for p in df_piatti["piatto"].tolist()]
    bm25 = BM25Okapi(tokenized_piatti)

    # Salvataggio su file
    with open(config.DATA_PATH + "preprocessed/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print(" Modello BM25 salvato con successo!")

# **Creazione e salvataggio della matrice TF-IDF**
def create_tfidf(df_piatti):
    """Crea e salva la matrice TF-IDF per migliorare il retrieval testuale."""
    print("📊 Creazione della matrice TF-IDF...")

    # Creazione del vettorizzatore TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_piatti["piatto"].tolist())

    # Salvataggio su file
    with open(config.DATA_PATH + "preprocessed/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print(" Matrice TF-IDF salvata con successo!")

# **Esecuzione della pipeline di preprocessing**
if __name__ == "__main__":
    print(" Avvio del processo di preprocessing...")
    
    # 1 Carica i dati
    df_piatti, dish_mapping, domande_data = load_data()

    #  Pulisce i dati
    df_piatti = clean_data(df_piatti)

    # Salva il dataset pulito
    df_piatti.to_csv(config.DATA_PATH + "preprocessed/piatti_cleaned.csv", index=False)

    #  Creazione e salvataggio degli indici
    create_faiss_index(df_piatti)
    create_bm25(df_piatti)
    create_tfidf(df_piatti)

    print(" Preprocessing completato! Tutti i dati sono stati salvati nella cartella `preprocessed/`")
