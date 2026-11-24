import ollama


class OllamaLLM:
    def __init__(self, host: str, model_name: str, temperature: float = 0.1, num_ctx: int = 4096):
        self._client = ollama.Client(host=host)
        self._model = model_name
        self._options = {"temperature": temperature, "num_ctx": num_ctx}

    def generate(self, system: str, prompt: str) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            options=self._options,
        )
        return response["message"]["content"]
