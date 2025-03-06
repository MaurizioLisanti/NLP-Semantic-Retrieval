"""
config.py
---------
Questo modulo contiene tutte le configurazioni globali della pipeline.

Funzionalità principali:
✅ Definisce i modelli NLP e LLM da utilizzare
✅ Specifica i parametri per retrieval e reranking
✅ Centralizza i percorsi dei file per evitare hardcoding

Autore: Maurizio
"""

import os

# ** Percorsi dei file**
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Percorso base del progetto
DATA_PATH = os.path.join(BASE_DIR, "data/")  # Cartella contenente i dataset
PREPROCESSED_PATH = os.path.join(DATA_PATH, "preprocessed/")  # Dati puliti e ottimizzati
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission/")  # Cartella per i file di submission

# **Configurazione dei modelli**
# Embedding Model → Utilizzato per FAISS
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Modello per il reranking con LLM
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# **Parametri di Retrieval**
TOP_K_RETRIEVAL = 10  # Numero di risultati da restituire dal retrieval (FAISS + BM25 + TF-IDF)
TOP_K_RERANKING = 5  # Numero di risultati finali da restituire dopo il reranking

# ** Percorsi dei file pre-processati**
PIATTI_CLEANED = os.path.join(PREPROCESSED_PATH, "piatti_cleaned.csv")  # Dataset pulito dei piatti
FAISS_INDEX = os.path.join(PREPROCESSED_PATH, "faiss_index.bin")  # Indice FAISS
BM25_INDEX = os.path.join(PREPROCESSED_PATH, "bm25_index.pkl")  # Modello BM25
TFIDF_MATRIX = os.path.join(PREPROCESSED_PATH, "tfidf_matrix.pkl")  # Matrice TF-IDF
EMBEDDING_VECTORS = os.path.join(PREPROCESSED_PATH, "embedding_vectors.npy")  # Embedding salvati

# ** Nome del file di submission**
SUBMISSION_FILE = os.path.join(SUBMISSION_PATH, "submission.csv")

# ** Logging e Debugging**
LOGGING_ENABLED = True  # Attiva o disattiva il logging per il debugging
LOG_FILE = os.path.join(BASE_DIR, "logs/pipeline.log")  # File di log

# **Funzione per la stampa dei log**
def log_message(message):
    """
    Funzione per stampare messaggi di log solo se il logging è attivato.

    Args:
        message (str): Messaggio da stampare.
    """
    if LOGGING_ENABLED:
        print(f" LOG: {message}")

# **Controllo delle cartelle**
def create_directories():
    """
    Crea le cartelle necessarie se non esistono già.
    """
    for path in [DATA_PATH, PREPROCESSED_PATH, SUBMISSION_PATH, os.path.dirname(LOG_FILE)]:
        os.makedirs(path, exist_ok=True)

# **Esegui la creazione delle cartelle all'importazione del modulo**
create_directories()
