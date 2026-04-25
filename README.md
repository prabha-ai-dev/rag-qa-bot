RAG(QAchain)bot

A local RAG-based QA system using LangChain, FAISS, and Flan-T5. Ask questions about your text documents — no API keys needed.

How it works
1. Loads a `.txt` file and splits it into chunks
2. Embeds chunks using `sentence-transformers/all-mpnet-base-v2`
3. Stores embeddings in FAISS vector store
4. Retrieves top-3 relevant chunks per query
5. Generates answers using `google/flan-t5-base`

Setup
```
pip install langchain langchain-community faiss-cpu sentence-transformers transformers torch
```

Tech Stack
| Component | Tool |
|---|---|
| Framework | LangChain |
| Embeddings | all-mpnet-base-v2 |
| Vector Store | FAISS |
| LLM | Flan-T5-base |

Limitations
- Small model — may give brief answers on complex queries
- Re-embeds document on every run
- Works best on focused, factual text
