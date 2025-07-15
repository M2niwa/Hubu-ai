import requests
from config.settings import settings

class DeepSeekClient:
    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.7):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.model = model
        self.temperature = temperature
        self.url = "https://api.deepseek.com/v1/chat/completions"

    def query(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(self.url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"DeepSeek API Error: {response.text}")
