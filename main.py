from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import requests
from typing import List, Dict, Any
from openai import OpenAI

# -----------------------------
# Load env + init clients
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="NVIDIA Chatbot (news-aware)")

# Allow local file (index.html) to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class EchoRequest(BaseModel):
    message: str

class QuestionRequest(BaseModel):
    question: str

class NewsQuery(BaseModel):
    query: str = "NVIDIA"
    days: int = 7
    page_size: int = 5
    language: str = "en"

# -----------------------------
# Tiny in-memory cache (demo)
# -----------------------------
_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"ts": float, "data": any}
CACHE_TTL_SECONDS = 180  # 3 minutes to avoid rate limits while demoing

def get_cache(key: str):
    item = _cache.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > CACHE_TTL_SECONDS:
        return None
    return item["data"]

def set_cache(key: str, data: Any):
    _cache[key] = {"ts": time.time(), "data": data}

# -----------------------------
# News helpers
# -----------------------------
def fetch_news(query: str = "NVIDIA", *, days: int = 7, page_size: int = 5, language: str = "en"):
    """
    Calls NewsAPI to get recent articles. Returns a list of dicts with title, url, source, publishedAt.
    """
    cache_key = f"news::{query}::{days}::{page_size}::{language}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    # NewsAPI 'from' parameter expects ISO date; to keep it simple we sort by publishedAt
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&language={language}"
        f"&sortBy=publishedAt"
        f"&pageSize={page_size}"
        f"&apiKey={NEWS_API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    articles = raw.get("articles", []) or []
    cleaned = []
    for a in articles:
        cleaned.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "source": (a.get("source") or {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "description": a.get("description"),
            "content": a.get("content"),
        })

    set_cache(cache_key, cleaned)
    return cleaned

def summarize_articles(articles: List[Dict[str, Any]], user_question: str) -> str:
    """
    Uses OpenAI to produce a short, factual, source-linked summary tailored to the user's question.
    """
    if not articles:
        return "No recent relevant articles found."

    # Build a compact context with titles + sources + URLs
    lines = []
    for i, a in enumerate(articles, start=1):
        title = a.get("title") or "Untitled"
        src = a.get("source") or "Unknown"
        url = a.get("url") or ""
        lines.append(f"{i}. {title} — {src} — {url}")

    context = "Recent NVIDIA-related articles:\n" + "\n".join(lines)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an NVIDIA news assistant. Answer succinctly and factually using only the info implied by the articles list. "
                    "If timing is uncertain, say so. Include 2–4 bullet points and end with a short 'Sources' list of the most relevant URLs."
                )
            },
            {
                "role": "user",
                "content": f"{context}\n\nUser question: {user_question}\n\nSummarize and answer:"
            }
        ],
        max_tokens=350,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()

def looks_like_news_intent(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "latest", "today", "this week", "recent", "news", "announce", "announced",
        "earnings", "quarter", "q1", "q2", "q3", "q4", "revenue", "guidance",
        "press release", "launch", "just released"
    ]
    return any(k in ql for k in keywords)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/echo")
async def echo(req: EchoRequest):
    return {"echo": req.message}

@app.post("/news")
async def news(q: NewsQuery):
    """
    Explicit news endpoint (handy for demos).
    """
    try:
        arts = fetch_news(q.query, days=q.days, page_size=q.page_size, language=q.language)
        summary = summarize_articles(arts, f"Summarize latest about {q.query}")
        return {"summary": summary, "articles": arts}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_nvidia(req: QuestionRequest):
    """
    Main Q&A. If the question looks time-sensitive, fetch + summarize news first, then answer.
    Otherwise, answer directly.
    """
    try:
        if looks_like_news_intent(req.question):
            # Fetch & summarize news, then answer
            arts = fetch_news("NVIDIA", days=7, page_size=6, language="en")
            news_summary = summarize_articles(arts, req.question)

            # Blend summary + final answer (keeps your assistant voice consistent)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an expert on NVIDIA, GPUs, AI hardware, and software. Be concise and accurate."},
                    {"role": "user",
                     "content": f"User question: {req.question}\n\nUse this news digest to answer:\n{news_summary}"}
                ],
                max_tokens=300,
                temperature=0.4,
            )
            answer = completion.choices[0].message.content.strip()
            return {"answer": answer, "mode": "news", "articles": arts}

        # Non-news question: answer directly
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are an expert on NVIDIA, GPUs, AI hardware, and software. Be clear and practical."},
                {"role": "user", "content": req.question}
            ],
            max_tokens=250,
            temperature=0.5,
        )
        answer = completion.choices[0].message.content.strip()
        return {"answer": answer, "mode": "direct"}

    except requests.HTTPError as http_err:
        return {"error": f"News API HTTP error: {http_err}"}
    except Exception as e:
        return {"error": str(e)}
