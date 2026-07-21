# 🚀 Free-Tier Deployment Plan (No OpenAI)

**Objective:** Deploy CHADUVU-GURU using **100% Free Tiers** and removing the heavy `sentence-transformers` dependency without using OpenAI.

## 🏗️ Architecture Overview

| Component | Current | Proposed Free Tier | Why? |
| :--- | :--- | :--- | :--- |
| **Frontend** | Localhost | **Vercel** | Free, fast global CDN, simple git push deploy. |
| **Backend** | Localhost | **Render** | Free tier for Web Services (Python/FastAPI). |
| **Database** | Qdrant (Local) | **Qdrant Cloud** | Free 1GB forever cluster. No credit card required. |
| **Embeddings** | `sentence-transformers` | **Gemini Embeddings** | Free with Google AI Studio key. Already using Gemini SDK. |
| **LLM** | Gemini | **Gemini** | Continues as is. |

---

## 🛠️ Step 1: Remove Heavy Dependencies

The `sentence-transformers` library downloads models (~100MB - 1GB+) which causes:
1.  **Slug Size Issues:** Exceeds limits on Vercel/Render free tiers.
2.  **RAM Issues:** Crashes on 512MB RAM instances (Render Free).
3.  **Startup Time:** Slow cold starts.

### **Solution: Use Gemini Embeddings**
Since you are already using `google-generativeai`, we can use the `embedding-001` or `text-embedding-004` model.

**Changes Required in `backend/qdrant.py`:**

```python
# REMOVE
# from sentence_transformers import SentenceTransformer
# local_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ADD
import google.generativeai as genai

# In initialize():
# No need to load local_embedder model!

# REPLACEMENT FUNCTION
def get_embedding(text: str) -> List[float]:
    result = genai.embed_content(
        model="models/text-embedding-004", # Latest optimized embedding model
        content=text,
        task_type="retrieval_document", 
        title="Embedding of text"
    )
    return result['embedding']
```

**Impact:**
- `sentence-transformers` removed from `requirements.txt`.
- Slug size reduces by ~500MB.
- RAM usage drops significantly.

---

## ☁️ Step 2: Deployment Setup

### **2.1 Database: Qdrant Cloud (Free)**
1.  Sign up at [cloud.qdrant.io](https://cloud.qdrant.io).
2.  Create a **Free Tier Cluster**.
3.  Get the **API URL** and **API KEY**.
4.  Update your environment variables.

### **2.2 Backend: Render (Free)**
1.  **Create `render.yaml`:**
    ```yaml
    services:
      - type: web
        name: chaduvu-guru-backend
        env: python
        buildCommand: pip install -r requirements.txt
        startCommand: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
        envVars:
          - key: PYTHON_VERSION
            value: 3.11.0
          - key: QDRANT_URL
            sync: false
          - key: QDRANT_API_KEY
            sync: false
          - key: GOOGLE_API_KEY
            sync: false
    ```
2.  Push to GitHub.
3.  Connect Render to your repo.

### **2.3 Frontend: Vercel (Free)**
1.  **Create `vercel.json`:**
    ```json
    {
      "rewrites": [
        { "source": "/api/(.*)", "destination": "https://your-render-backend.onrender.com/api/$1" }
      ]
    }
    ```
2.  Push to GitHub.
3.  Import project in Vercel.

---

## 📝 Implementation Steps

1.  **Modify Code**: Replace `sentence-transformers` with `genai.embed_content` in `backend/qdrant.py` and `backend/intent_classifier.py`.
2.  **Update Config**: Remove `sentence-transformers` from `requirements.txt`.
3.  **Verify**: Run locally to ensure Gemini embeddings work (they are 768 dimensions usually, whereas MiniLM is 384, so **re-indexing is required**).
4.  **Push & Deploy**: Push changes and deploy to platforms.

## ⚠️ Important Note on Data
Switching embedding models (MiniLM -> Gemini) changes the vector dimensions. **You MUST clear your Qdrant collection and re-upload all books.** The code supports this via `process_and_embed_book`.

---

**Next Steps:**
Shall we proceed with **replacing `sentence-transformers` with Gemini Embeddings** in the code?
