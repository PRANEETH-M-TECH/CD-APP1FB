# 🚀 CHADUVU-GURU Deployment Strategy Guide

**Created:** December 5, 2025  
**Status:** Planning Document - No Implementation Yet  
**Purpose:** Evaluate all deployment options for production launch

---

## 📊 **Executive Summary**

### **Current Blockers:**
- ❌ Vercel doesn't support `uvicorn` (FastAPI)
- ❌ `sentence-transformers` library is **~2GB** (exceeds most free tier limits)
- ❌ Heavy ML models not suitable for serverless platforms

### **Solutions Evaluated:**
1. ✅ **Split Deployment** (Frontend + Backend separately)
2. ✅ **Embedding API Replacement** (Sentence-transformers → OpenAI/Cohere)
3. ✅ **Full-Stack Free Hosting** (Render, Railway, Fly.io)

---

## 🎯 **Option 1: Split Frontend & Backend Deployment (Recommended)**

### **Why This Approach?**
- ✅ Leverage free tiers of multiple providers
- ✅ Frontend on fast CDN (Vercel/Netlify)
- ✅ Backend on Python-friendly platform (Render/Railway)
- ✅ **Both support branch-based deployments**

---

### **1.1 Frontend Deployment: Vercel (FREE)**

#### **Platform:** [Vercel](https://vercel.com)
#### **Cost:** **$0/month** (Generous free tier)

#### **Features:**
- ✅ Unlimited deployments
- ✅ **Branch-based deployments** (Deploy any Git branch)
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Preview URLs for each branch

#### **Setup Steps:**

```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Login
vercel login

# 3. Deploy specific branch (e.g., "dashboards")
git checkout dashboards
vercel --prod

# Or deploy with specific settings
vercel --prod --branch dashboards
```

#### **Configuration Required:**

Create `vercel.json` in project root:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "public/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/public/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/public/$1"
    }
  ]
}
```

#### **Environment Variables:**
```bash
BACKEND_API_URL=https://your-backend.onrender.com
```

#### **Cost:** **$0/month**
#### **Limits:**
- ✅ 100GB bandwidth/month
- ✅ 100 deployments/day
- ✅ Unlimited static sites

---

### **1.2 Backend Deployment: Render (FREE)**

#### **Platform:** [Render](https://render.com)
#### **Cost:** **$0/month** (Free tier with limitations)

#### **Features:**
- ✅ Native Python/FastAPI support
- ✅ **Branch-based auto-deploys**
- ✅ Automatic HTTPS
- ✅ PostgreSQL/Redis free tiers
- ✅ Background jobs support

#### **⚠️ Free Tier Limitations:**
- ❌ **Spins down after 15 minutes of inactivity** (cold start: ~30 seconds)
- ✅ 750 hours/month (enough for 1 service 24/7)
- ✅ 512MB RAM
- ✅ 0.1 CPU

#### **Setup Steps:**

1. **Create `render.yaml` in project root:**

```yaml
services:
  - type: web
    name: chaduvu-guru-api
    env: python
    region: singapore  # Choose closest to India
    branch: dashboards  # SPECIFY YOUR BRANCH HERE
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.14
      - key: GOOGLE_API_KEY
        sync: false  # Set in Render dashboard
      - key: QDRANT_URL
        value: https://your-qdrant-cloud.io
    autoDeploy: true  # Auto-deploy on git push
```

2. **Connect GitHub Repo:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - **Select branch:** `dashboards` (or any branch you want)
   - Render will auto-detect `render.yaml`

3. **Set Environment Variables in Dashboard:**
   ```
   GOOGLE_API_KEY=your_gemini_key
   QDRANT_URL=your_qdrant_cloud_url
   FIREBASE_CREDENTIALS=<paste JSON>
   ```

#### **Branch Deployment:**
- Render lets you deploy **specific branches**
- Each branch can have its own service
- Example:
  - `main` branch → `chaduvu-guru-prod`
  - `dashboards` branch → `chaduvu-guru-staging`

#### **Cost:** **$0/month**
#### **Paid Upgrade:** $7/month removes sleep (always online)

---

### **1.3 Alternative Backend: Railway (FREE)**

#### **Platform:** [Railway](https://railway.app)
#### **Cost:** **$5 free credit/month** (No credit card required)

#### **Features:**
- ✅ No sleep/cold starts (better than Render free)
- ✅ **Branch-based deployments**
- ✅ Automatic HTTPS
- ✅ 1GB RAM free tier
- ✅ PostgreSQL/Redis included

#### **Setup:**

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Deploy specific branch
git checkout dashboards
railway up
```

#### **Branch Deployment:**
Railway supports **GitHub integration** with branch selection:
- Go to Railway dashboard
- Connect GitHub
- Choose repository
- **Select branch to deploy** (e.g., `dashboards`)
- Auto-deploys on push

#### **Cost:** **$5 credit/month** (Usually enough for small apps)
#### **Paid:** $0.000231/GB/sec usage (pay-as-you-go)

---

### **1.4 Updated Frontend API Calls**

After deploying backend, update all frontend API calls:

**Old (Local):**
```javascript
fetch('/api/smart_query?...')
```

**New (Production):**
```javascript
const BACKEND_URL = 'https://chaduvu-guru.onrender.com';

fetch(`${BACKEND_URL}/api/smart_query?...`)
```

**Better: Use Environment Variable**

In `public/index.html` and all pages:
```javascript
const BACKEND_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : 'https://chaduvu-guru.onrender.com';
```

---

## 🔄 **Option 2: Embedding API Replacements**

### **Problem:**
`sentence-transformers` requires:
- ✅ ~2GB model download
- ❌ Not suitable for serverless
- ❌ Slow cold starts

### **Solution: Use Cloud Embedding APIs**

---

### **2.1 OpenAI Embeddings (RECOMMENDED)**

#### **Model:** `text-embedding-3-small` (Latest, cheapest)

#### **Pricing:**
```
Cost Per 1M Tokens: $0.02
Tokens per query: ~200 (average question)
Cost per query: $0.000004 (0.0004 cents)
```

#### **Monthly Cost Estimate:**

| Queries/Month | Cost/Month |
|---------------|------------|
| 10,000 | $0.04 |
| 50,000 | $0.20 |
| 100,000 | $0.40 |
| 500,000 | $2.00 |
| 1,000,000 | $4.00 |

#### **💰 Real-World Example:**
- **100 students** × **50 queries/month** = **5,000 queries**
- **Cost:** **$0.02/month** (2 cents!)

#### **Implementation:**

```python
# Replace sentence-transformers with OpenAI
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dimensions
        input=text
    )
    return response.data[0].embedding
```

#### **Benefits:**
- ✅ **Zero deployment** (cloud API)
- ✅ **Fast** (no model loading)
- ✅ **Extremely cheap**
- ✅ **High quality** (better than sentence-transformers)

#### **Cons:**
- ❌ Requires internet (not offline)
- ❌ Dependency on OpenAI

---

### **2.2 Cohere Embeddings (Alternative)**

#### **Model:** `embed-english-v3.0`

#### **Pricing:**
```
Cost Per 1M Tokens: $0.10
~5x more expensive than OpenAI
```

#### **Monthly Cost Estimate:**

| Queries/Month | Cost/Month |
|---------------|------------|
| 10,000 | $0.20 |
| 50,000 | $1.00 |
| 100,000 | $2.00 |

#### **Benefits:**
- ✅ Good quality
- ✅ Free trial credits

#### **Cons:**
- ❌ More expensive than OpenAI
- ❌ Smaller community

---

### **2.3 Voyage AI Embeddings (Best Quality)**

#### **Model:** `voyage-2`

#### **Pricing:**
```
Cost Per 1M Tokens: $0.12
```

#### **Benefits:**
- ✅ **Best quality** (MTEB leaderboard)
- ✅ Optimized for retrieval

#### **Cons:**
- ❌ Most expensive option

---

### **💡 Recommendation:**

Use **OpenAI `text-embedding-3-small`**:
- ✅ Cheapest ($0.02 per 1M tokens)
- ✅ Best deployment (no heavy models)
- ✅ Industry standard
- ✅ Fast and reliable

**Estimated cost for small school (1000 students):**
- 1000 students × 100 queries/month = **100,000 queries**
- **Cost: $0.40/month** (40 cents!)

---

## 🏗️ **Option 3: Full-Stack Deployment (Single Platform)**

### **3.1 Fly.io (RECOMMENDED for Full-Stack)**

#### **Platform:** [Fly.io](https://fly.io)
#### **Cost:** **$0-5/month**

#### **Features:**
- ✅ Full FastAPI + Frontend support
- ✅ **Branch-based deployments**
- ✅ Persistent volumes (for uploads)
- ✅ Global CDN
- ✅ PostgreSQL/Redis free tier

#### **Free Tier:**
- ✅ 3 VMs (256MB RAM each)
- ✅ 3GB persistent storage
- ✅ 160GB bandwidth/month

#### **Setup:**

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Login
flyctl auth login

# 3. Initialize
flyctl launch

# 4. Configure fly.toml
```

**fly.toml:**
```toml
app = "chaduvu-guru"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8000"
  PYTHON_VERSION = "3.11"

[[services]]
  http_checks = []
  internal_port = 8000
  processes = ["app"]
  protocol = "tcp"

  [services.concurrency]
    hard_limit = 25
    soft_limit = 20

  [[services.ports]]
    force_https = true
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

#### **Branch Deployment:**
```bash
# Deploy specific branch
git checkout dashboards
flyctl deploy --build-arg BRANCH=dashboards
```

#### **Cost:** **$0-5/month** (depends on usage)

---

### **3.2 Render Full-Stack**

Deploy both frontend and backend on Render:
- **Static Site** (Frontend): $0/month
- **Web Service** (Backend): $0/month (with sleep) or $7/month (always on)

---

## 📈 **Cost Comparison Summary**

### **Split Deployment (Frontend + Backend)**

| Component | Platform | Cost | Notes |
|-----------|----------|------|-------|
| Frontend | Vercel | $0 | Unlimited |
| Backend | Render | $0 ($7 no-sleep) | 15min sleep |
| Embeddings | OpenAI | $0.40/month | 100K queries |
| **TOTAL** | | **$0.40-$7.40** | |

### **Full-Stack Deployment**

| Component | Platform | Cost | Notes |
|-----------|----------|------|-------|
| App + DB | Fly.io | $0-$5 | Free tier |
| Embeddings | OpenAI | $0.40/month | 100K queries |
| **TOTAL** | | **$0.40-$5.40** | |

---

## ✅ **Recommended Deployment Strategy**

### **For Your Use Case:**

#### **Phase 1: Replace Embeddings (Critical)**

```python
# In backend/qdrant.py or embeddings.py

# OLD (sentence-transformers)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)

# NEW (OpenAI)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

**Cost Impact:** ~$0.40/month for 100K queries (negligible)

---

#### **Phase 2: Deploy Backend to Render**

1. Create `render.yaml` (see section 1.2)
2. Connect GitHub repo to Render
3. **Select `dashboards` branch**
4. Set environment variables
5. Deploy

**Cost:** $0/month (or $7/month for no-sleep)

---

#### **Phase 3: Deploy Frontend to Vercel**

1. Create `vercel.json` (see section 1.1)
2. Connect GitHub repo to Vercel
3. **Select `dashboards` branch**
4. Set `BACKEND_API_URL` environment variable
5. Deploy

**Cost:** $0/month

---

#### **Phase 4: Update API Calls**

Update all frontend files to use `BACKEND_API_URL`:

```javascript
const API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';

fetch(`${API_URL}/api/smart_query?...`)
```

---

## 🎯 **Final Recommendations**

### **Best Approach for CHADUVU-GURU:**

1. ✅ **Replace `sentence-transformers` with OpenAI embeddings**
   - Cost: ~$0.40/month (100K queries)
   - Deployment-friendly
   - Better quality

2. ✅ **Split Deployment:**
   - **Frontend:** Vercel (Free)
   - **Backend:** Render (Free or $7/month)

3. ✅ **Branch Selection:**
   - Both Vercel and Render support branch-based deployments
   - Deploy `dashboards` branch for testing
   - Deploy `main` branch for production

4. ✅ **Total Monthly Cost:**
   - **Cheapest:** $0.40/month (Render free + OpenAI)
   - **Recommended:** $7.40/month (Render paid + OpenAI)
   - **With Fly.io:** $5.40/month (Fly + OpenAI)

---

## 📝 **Implementation Checklist**

### **Pre-Deployment:**

- [ ] Replace sentence-transformers with OpenAI embeddings
- [ ] Test locally with OpenAI API
- [ ] Create `render.yaml` for backend
- [ ] Create `vercel.json` for frontend
- [ ] Update API URLs in frontend
- [ ] Set up environment variables

### **Backend Deployment:**

- [ ] Connect Render to GitHub
- [ ] Select `dashboards` branch
- [ ] Configure environment variables
- [ ] Deploy and test

### **Frontend Deployment:**

- [ ] Connect Vercel to GitHub
- [ ] Select `dashboards` branch
- [ ] Set `BACKEND_API_URL`
- [ ] Deploy and test

### **Post-Deployment:**

- [ ] Test end-to-end functionality
- [ ] Monitor costs (OpenAI dashboard)
- [ ] Set up alerts for errors
- [ ] Create production deployment (main branch)

---

## 🚨 **Risks & Mitigation**

### **Risk 1: API Costs Spike**

**Mitigation:**
- Set OpenAI spending limits ($10/month)
- Monitor usage daily
- Implement query caching
- Rate limit users

### **Risk 2: Backend Cold Starts (Render Free)**

**Mitigation:**
- Upgrade to $7/month (no sleep)
- OR use Railway ($5 credit)
- OR implement health check pings

### **Risk 3: CORS Issues (Frontend ↔ Backend)**

**Mitigation:**
```python
# In backend/app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 **Competitor Comparison**

| Platform | Free Tier | Branch Deploy | Python Support | Cold Start | Best For |
|----------|-----------|---------------|----------------|------------|----------|
| **Vercel** | ✅ Generous | ✅ Yes | ❌ No | N/A | Frontend |
| **Render** | ✅ Limited | ✅ Yes | ✅ Yes | 30s | Backend |
| **Railway** | ✅ $5 credit | ✅ Yes | ✅ Yes | None | Backend |
| **Fly.io** | ✅ Good | ✅ Yes | ✅ Yes | <1s | Full-Stack |
| **Heroku** | ❌ Paid only | ✅ Yes | ✅ Yes | Fast | Legacy |

---

## 🎓 **Learning Resources**

### **Deployment Guides:**
- [Render FastAPI Guide](https://render.com/docs/deploy-fastapi)
- [Vercel Static Sites](https://vercel.com/docs/frameworks/static-sites)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)

### **Cost Calculators:**
- [OpenAI Pricing Calculator](https://openai.com/api/pricing/)
- [Render Pricing](https://render.com/pricing)
- [Railway Pricing](https://railway.app/pricing)

---

## ✅ **Next Steps**

1. **Review this document** and choose your approach
2. **Test OpenAI embeddings locally** (critical change)
3. **Create deployment configs** (render.yaml, vercel.json)
4. **Deploy to staging** (dashboards branch)
5. **Test thoroughly**
6. **Deploy to production** (main branch)

---

**Document Status:** ✅ Ready for Review  
**No Implementation Done:** This is a planning document only  
**Awaiting Your Decision:** Choose deployment approach and proceed

---

**Questions?** Review each section and let me know which approach you want to implement!
