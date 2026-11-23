import httpx

from app.config import get_settings

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

settings = get_settings()
body = {
    "content": {"parts": [{"text": "مدة الإشعار عند الاستقالة ثلاثون يوماً"}]},
    "taskType": "RETRIEVAL_DOCUMENT",
    "outputDimensionality": 768,
}
response = httpx.post(URL, json=body, headers={"x-goog-api-key": settings.gemini_api_key}, timeout=30)
print("status:", response.status_code)
if response.status_code == 200:
    values = response.json()["embedding"]["values"]
    print("vector length:", len(values), "first values:", [round(v, 4) for v in values[:3]])
else:
    print(response.text[:500])