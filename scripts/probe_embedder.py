import math

from app.config import get_settings
from app.infrastructure.gemini_embedder import GeminiEmbedder

DOCS = [
    "مدة الإشعار عند الاستقالة ثلاثون يوماً ويجب تقديمها كتابياً للإدارة.",
    "تبدأ ساعات الدوام الرسمي من الثامنة صباحاً حتى الرابعة مساءً من الأحد إلى الخميس.",
    "رسوم التأسيس لباقة Growth تبلغ 900 USD والاشتراك الشهري 220 USD.",
    "The capital of France is Paris and it is known for the Eiffel Tower.",
]
QUESTIONS = [
    ("كم مدة الإشعار عند الاستقالة؟", 0),
    ("متى يبدأ الدوام الرسمي؟", 1),
    ("كم سعر باقة النمو؟", 2),
    ("how many days of notice are needed to resign?", 0),
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


embedder = GeminiEmbedder(get_settings().gemini_api_key)
doc_vectors = embedder.embed_texts(DOCS)
print("vectors:", len(doc_vectors), "x", len(doc_vectors[0]))
for number, (question, expected) in enumerate(QUESTIONS, start=1):
    query_vector = embedder.embed_query(question)
    scores = [cosine(query_vector, v) for v in doc_vectors]
    best = scores.index(max(scores))
    verdict = "OK" if best == expected else "WRONG"
    print(f"Q{number}: best={best} expected={expected} {verdict} scores={[round(s, 2) for s in scores]}")