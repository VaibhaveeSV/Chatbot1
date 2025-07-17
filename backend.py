from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import cohere

app = FastAPI()

co = cohere.Client("mW8xyq8DQ8UBZErjphQJNOvFqZ6UKlP5II3wgxMO") 
emb = SentenceTransformer("all-MiniLM-L6-v2")

faqs = [
    {"q": "How often are team meetings held?", "a": "We hold weekly standups and monthly knowledge-sharing sessions."},
    {"q": "Can I contribute to open-source projects?", "a": "Yes, employees and interns are encouraged to contribute to OSS."},
    {"q": "What are embeddings?", "a": "Embeddings are vector representations of text used in search and NLP."},
    {"q": "How do I get API access?", "a": "You can request API keys by filling out the internal developer form."},
    {"q": "Is there a document for onboarding?", "a": "Yes, new members receive a PDF guide and a Notion onboarding page."},
    {"q": "What is Cohere used for?", "a": "We use Cohere for both text generation and semantic embeddings."},
    {"q": "How do I join the internal Slack?", "a": "You'll receive an invite to Slack after HR onboarding is complete."},
    {"q": "What are the core values of Resolute.ai?", "a": "Curiosity, innovation, integrity, and collaboration."},
    {"q": "How do I raise a bug or feature request?", "a": "Use the internal issue tracker or report via Slack to the dev team."}
]


answers = [faq["a"] for faq in faqs]
ans_emb = emb.encode(answers)
index = faiss.IndexFlatL2(len(ans_emb[0]))
index.add(np.array(ans_emb).astype("float32"))

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: QuestionRequest):
    question = request.question
    que_emb = emb.encode(question)
    _, I = index.search(np.array([que_emb]).astype("float32"), k=1)
    context = answers[I[0][0]]
    prompt = f"Use this context to answer:\n\n{context}\n\nQuestion: {question}"
    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=300
    )
    answer = response.generations[0].text.strip()

    return {"answer": answer}