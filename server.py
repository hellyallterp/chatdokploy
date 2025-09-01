# server.py
from fastapi import FastAPI
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# 1️⃣  Il modello da usare (puoi cambiare il nome più avanti in Dokploy)
MODEL = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    device_map="auto"
)

@app.post("/chat")
async def chat(body: dict):
    prompt = body.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=200)
    return {"response": tokenizer.decode(out[0], skip_special_tokens=True)}

# 2️⃣  Una semplice pagina web
@app.get("/")
async def home():
    with open("index.html") as f:
        return f.read()
