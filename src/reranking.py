"""
reranking.py
------------
Modulo per il reranking dei risultati ottenuti dal retrieval con FAISS, BM25 e TF-IDF.

Funzionalità principali:
✅ Usa un LLM per riorganizzare i risultati in base alla pertinenza con la query
✅ Modello di riferimento: `cross-encoder/ms-marco-MiniLM-L-12-v2`
✅ Filtra e restituisce solo i risultati più rilevanti

Autore: [Maurizio]

"""

from sentence_transformers import CrossEncoder
import config

# ** Caricamento del modello di reranking**
def load_reranker():
    """
    Carica il modello di Cross-Encoder per il reranking.

    Returns:
        CrossEncoder: Modello di reranking pre-addestrato.
    """
    print(" Caricamento del modello di reranking...")
    reranker = CrossEncoder(config.RERANKER_MODEL)
    print(" Modello di reranking caricato con successo!")
    return reranker

# **Funzione per il reranking**
def rerank_results(query, candidates, reranker, top_k=5):
    """
    Esegue il reranking dei risultati in base alla pertinenza con la query.

    Args:
        query (str): La query dell'utente.
        candidates (list): Lista dei piatti trovati dal retrieval.
        reranker (CrossEncoder): Modello di LLM per valutare la pertinenza.
        top_k (int): Numero di risultati finali da restituire.

    Returns:
        list: Lista dei migliori piatti ordinati in base alla pertinenza.
    """
    if not candidates:
        print(" Nessun candidato da rerankare.")
        return []

    print(f" Reranking di {len(candidates)} risultati per la query: {query}")

    # **Creazione delle coppie (query, piatto)**
    candidate_pairs = [(query, p) for p in candidates]

    # **Assegna uno score di pertinenza a ogni coppia**
    scores = reranker.predict(candidate_pairs)

    # **Ordina i risultati in base allo score**
    sorted_candidates = [candidates[i] for i in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)]

    # **Restituisce solo i migliori `top_k` risultati**
    return sorted_candidates[:top_k]

# **Esecuzione del reranking**
if __name__ == "__main__":
    print("Avvio del processo di reranking...")

    # Esempio di input (da sostituire con dati reali dal retrieval)
    query = "Quali piatti contengono tartufo?"
    retrieved_results = ["Risotto al tartufo", "Pizza ai funghi", "Pasta al tartufo", "Gnocchi alla crema di funghi"]

    # Carica il modello di reranking
    reranker = load_reranker()

    # Esegue il reranking
    ranked_results = rerank_results(query, retrieved_results, reranker)

    print(f" Risultati ordinati: {ranked_results}")
