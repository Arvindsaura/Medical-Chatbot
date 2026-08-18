import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
models = r.json()["data"]
for m in models:
    if m.get("input_modalities") == ["text"] and m.get("output_modalities") == ["text"]:
        print(m["id"], "|", m["name"])
