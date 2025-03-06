#AI-Powered Information Retrieval System

##  Introduzione
Questo progetto è stato sviluppato per la competizione Kaggle "Ciclo Cosmico 789".  
L'obiettivo era creare un **sistema avanzato di retrieval basato su NLP** per suggerire piatti intergalattici in base alle richieste degli utenti.

La soluzione combina **tecniche di retrieval semantico e testuale** con **un sistema di reranking basato su LLM** per migliorare la precisione dei risultati.

---

## 📂 Struttura del Progetto
📂 `hackathon_project/`  
│── 📂 `data/` → Contiene i dataset grezzi e pre-elaborati  
│    ├── `database_piatti_con_id.csv`  
│    ├── `dish_mapping.json`  
│    ├── `domande.json`  
│    ├── `submission.csv`  
│    ├── 📂 `preprocessed/` → Dati puliti e ottimizzati  
│── 📂 `models/` → Contiene modelli NLP e LLM ottimizzati  
│    ├── `embedding_model/` → Modelli di embedding (`bge-large`, `mpnet-base-v2`)  
│    ├── `reranker_model/` → Modelli per il reranking (`cross-encoder/ms-marco-MiniLM-L-12-v2`)  
│    ├── `faiss_index/` → File dell’indice FAISS pre-addestrato  
│── 📂 `src/` → Contiene il codice della pipeline  
│    ├── `preprocessing.py` → Pulizia e preparazione dati  
│    ├── `retrieval.py` → FAISS + BM25 + TF-IDF  
│    ├── `reranking.py` → Ordinamento delle risposte con LLM  
│    ├── `generate_submission.py` → Pipeline completa per la submission  
│    ├── `config.py` → Configurazioni globali (modelli, top_k, path, etc.)  
│── 📂 `notebooks/` → Contiene Jupyter Notebook per analisi ed esperimenti  
│── 📂 `logs/` → Contiene log per il debugging  
│── 📂 `submission/` → Cartella con il file `submission.csv`  
│── 📄 `requirements.txt` → Librerie necessarie  
│── 📄 `README.md` → Documentazione del progetto  

---

##  Tecnologie Utilizzate
 **FAISS** → Per la ricerca veloce basata su similarità semantica.  
 **BM25** → Per il retrieval basato su parole chiave.  
 **TF-IDF** → Per il miglioramento della ricerca tra documenti simili.  
 **Sentence Transformers** → Per generare embedding NLP avanzati.  
 **Cross-Encoder LLM** → Per il reranking basato su IA.  
 **Pandas, Scikit-learn, NumPy** → Per la gestione e analisi dei dati.  

---

##  Stato del Progetto
 **Retrieval funzionante con FAISS + BM25 + TF-IDF**  
 **Pipeline ben strutturata e documentata**  
 **Il reranking con LLM è implementato ma non ottimizzato**  
 **Alcuni test avanzati sugli embedding non sono stati completati**  

N.B. **Il Progetto da completare.**  

---

##  Come Eseguire il Codice
**Installare le librerie necessarie:**  
```bash
pip install -r requirements.txt
