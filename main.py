from fastapi import FastAPI
from text_generation import Client

app = FastAPI()

@app.post("/generate")
async def generate(prompt: str):
    client = Client(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    output = client.generate(question=prompt, max_new_tokens=64)
    return {"response": output.generated_text}
