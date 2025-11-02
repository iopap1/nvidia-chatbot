from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from openai import OpenAI

# Initialize app
app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html (frontend)
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    try:
        # News API integration
        news_response = requests.get(
            f"https://newsapi.org/v2/everything?q=NVIDIA&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        news_titles = [a["title"] for a in news_response.json().get("articles", [])[:5]]
        news_summary = "\n".join(news_titles)

        # ChatGPT answer
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a chatbot that answers about NVIDIA."},
                {"role": "user", "content": f"{question}\nHere are the latest NVIDIA-related news headlines:\n{news_summary}"}
            ]
        )

        answer = completion.choices[0].message.content
        return JSONResponse(content={"answer": answer, "used_news": news_titles})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# Optional health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

