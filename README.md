# NVIDIA Chatbot  

A specialized ChatGPT-like assistant focused on **NVIDIA** — built with **FastAPI**, **OpenAI API**, and **NewsAPI**, and deployed on **Render**.  
It provides AI-powered answers and integrates real-time NVIDIA news headlines.

---

## Live Demo
**Try it here:** [https://nvidia-chatbot.onrender.com](https://nvidia-chatbot.onrender.com)

---

## Features
- Interactive chatbot that answers questions about NVIDIA
- Fetches and summarizes the latest NVIDIA-related news using NewsAPI
- Backend powered by FastAPI + OpenAI’s GPT-4 model
- Clean, responsive frontend built with HTML, CSS, and JavaScript
- Fully deployed online via Render

---

## Tech Stack
**Frontend:**  
- HTML  
- CSS  
- JavaScript  

**Backend:**  
- Python  
- FastAPI  
- OpenAI API  
- NewsAPI  

**Deployment:**  
- Render  

---

## Local Setup

If you’d like to run it locally:

```bash
# 1️⃣ Clone this repository
git clone https://github.com/iopap1/nvidia-chatbot.git
cd nvidia-chatbot

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add your API keys (create a .env file)
OPENAI_API_KEY=your_openai_api_key
NEWS_API_KEY=your_newsapi_key

# 4️⃣ Run the app
uvicorn main:app --reload
