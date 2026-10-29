from typing import List, Dict, Any
from task_base import Task
import json, random, re

class TaskMath(Task):
    def __init__(self):
        with open("prompts/math/math_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/math/math_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "gen"

    def get_dataset_file(self) -> str:
        return "data/sharded_instructions_600.json"

    def get_samples(self, filter="full"):
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        return data

    def get_task_name(self):
        return "math"
    
    def get_answer_description(self) -> str:
        return "The answer should be a single number (it could be decimal, or negative, or a fraction, etc.)."
    
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:

        regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]

        gold = sample["answer"].split("####")[1].strip().lower()

        try:

            extracted_answer = extracted_answer.strip()

            extracted_answer = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", extracted_answer)[-1]
            extracted_answer = [m for m in extracted_answer if m][0]
        except:
            return {"score": 0.0, "error": f"Answer could not be extracted: {repr(extracted_answer)}"}

        if re.search(r'\$', extracted_answer) and extracted_answer.endswith(".00"):
            extracted_answer = extracted_answer.rstrip(".00")                

        for regex in regexes_to_ignore:
            extracted_answer = re.sub(regex, "", extracted_answer)
            gold = re.sub(regex, "", gold)
        score = 1.0 if extracted_answer == gold else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.fully_specified_prompt.replace("[[QUESTION]]", sample["question"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = ""
        for shard in sample["shards"]:
            query += f"- {shard['shard']}\n"
        return self.fully_specified_prompt.replace("[[QUESTION]]", query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:

        return response["answer"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": sample["task_id"],
            "question": sample["question"],
            "answer": sample["answer"],
        }
