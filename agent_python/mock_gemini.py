# mock_gemini.py
from fastapi import FastAPI
from pydantic import BaseModel
import hashlib, math

app = FastAPI()

class EmbReq(BaseModel):
    input: str

class GenReq(BaseModel):
    prompt: str

@app.post("/embed")
def embed(req: EmbReq):
    s = req.input or ""
    # naive deterministic pseudo-embedding: convert sha256 bytes into floats
    h = hashlib.sha256(s.encode()).digest()
    vec = []
    for i in range(0, 64, 4):
        val = int.from_bytes(h[i:i+4], 'big', signed=False)
        vec.append(((val % 1000) / 1000.0) * (1 + (i/64)))
    return {"embedding": vec}

@app.post("/generate")
def gen(req: GenReq):
    p = req.prompt or ""
    # Very simple generator: echo first 300 chars + friendly sentence
    out = f"MOCK-GEMINI-REPLY: {p[:300]}...\n\n(You are using the local mock Gemini server.)"
    return {"output": out}
