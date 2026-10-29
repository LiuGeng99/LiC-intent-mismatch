from pathlib import Path

from model_openai import generate

ROOT = Path(__file__).resolve().parent

class Mediator:

    def __init__(self, task, model="deepseek-v3.2-thinking", prompt_path=None):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()
        path = Path(prompt_path) if prompt_path else ROOT / "prompts" / "mediator.txt"
        self.prompt = path.read_text()

    def generate_response(self, conversation, sample, temperature=1.0):
        transcript = ""
        for msg in conversation:
            if msg["role"] == "user":
                transcript += f"[User]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                transcript += f"[AI]: {msg['content']}\n"

        message = self.prompt.replace("[[CONVERSATION]]", transcript)
        response = generate(
            [{"role": "user", "content": message}],
            model=self.model,
            temperature=temperature,
            return_metadata=True,
        )
        return response["message"], response["total_usd"]
