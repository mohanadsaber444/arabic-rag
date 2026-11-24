import ollama


class OllamaEmbedder:
    def __init__(self, host: str, model_name: str, batch_size: int = 32):
        self._client = ollama.Client(host=host)
        self.model_name = model_name
        self._batch_size = batch_size
        self.dimension = len(self.embed_query("probe"))  # learned from the model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._client.embed(model=self.model_name, input=batch)["embeddings"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed(model=self.model_name, input=[text])["embeddings"][0]
