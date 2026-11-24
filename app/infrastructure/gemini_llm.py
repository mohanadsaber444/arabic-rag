from google import genai
from google.genai import types


class GeminiLLM:
    def __init__(
        self, api_key: str, model_name: str, temperature: float = 0.1, max_tokens: int = 2048
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name
        self._config = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens
        )

    def generate(self, system: str, prompt: str) -> str:
        config = self._config.model_copy(update={"system_instruction": system})
        response = self._client.models.generate_content(
            model=self._model, contents=prompt, config=config
        )
        return response.text or ""
