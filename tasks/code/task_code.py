from typing import List, Dict, Any, Tuple
import json
import random
import re
import pickle
import zlib
import base64
import ast

from task_base import Task
from tasks.code.eval_code import check_correctness

class TaskCode(Task):
    def __init__(self):
        with open("prompts/lcb/lcb_full_prompt.txt", "r") as f:
            self.fully_specified_prompt_lcb = f.read()
        with open("prompts/lcb/lcb_system_prompt.txt", "r") as f:
            self.system_prompt_lcb = f.read()
        with open("prompts/humaneval/humaneval_full_prompt.txt", "r") as f:
            self.fully_specified_prompt_humaneval = f.read()
        with open("prompts/humaneval/humaneval_system_prompt.txt", "r") as f:
            self.system_prompt_humaneval = f.read()

        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "task_specific"

    def get_dataset_file(self) -> str:
        return "data/sharded_instructions_600.json"

    def get_samples(self) -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        data = [d for d in data if d["task"] == "code"]
        return data

    def get_task_name(self):
        return "code"

    def get_answer_description(self) -> str:
        return ("A final answer must be a valid Python function that is defined (def ...) and returns "
                "something. The function should handle multiple test cases according to the problem specification. "
                "If the answer is just a natural language explanation, it is not a valid answer. "
                "Special symbols like newlines and quotes must be escaped.")

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt_lcb

    def load_test_cases(self, sample):
        public_test_cases = json.loads(sample["public_test_cases"])  # type: ignore

        if "private_test_cases" in sample:
            try:
                private_test_cases = json.loads(sample["private_test_cases"])  # type: ignore
            except:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(sample["private_test_cases"].encode("utf-8"))  # type: ignore
                        )
                    )
                )  # type: ignore
        else:
            private_test_cases = []

        return json.dumps(
            {
                "inputs": [
                    t["input"]
                    for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t["output"]
                    for t in public_test_cases + private_test_cases
                ],
                "fn_name": sample["metadata"].get("func_name", None),
            }
        )

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        pred_python_code = extracted_answer.replace("```python", "").replace("```", "")

        if "def " not in pred_python_code:
            return {"is_correct": False, "pass@1": 0, "score": 0}

        if "prompt" in sample:

            prompt_ast = ast.parse(sample["prompt"])
            imports = []
            for node in prompt_ast.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))

            if imports:
                pred_python_code = "\n".join(imports) + "\n\n" + pred_python_code

        old_func_name = pred_python_code.split("def ")[1].split("(")[0].strip()
        pred_python_code = pred_python_code.replace(old_func_name, sample["metadata"]["func_name"])

        testcases = self.load_test_cases(sample)

        output, metadata = check_correctness(sample, pred_python_code, testcases, timeout=6)
        all_test_cases_passed = all(o is True for o in output)

        score = 1 if all_test_cases_passed else 0
        return {"is_correct": all_test_cases_passed, "pass@1": 1 if all_test_cases_passed else 0, "score": score}

    def get_formatting_preamble(self, sample: Dict[str, Any]) -> Tuple[str, str]:

        call_preamble = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

        return call_preamble, f"```python\n{sample['starter_code'] if sample['starter_code'] else '# YOUR CODE HERE'}\n```"

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        if sample.get("source") in ["lcb_easy", "lcb_medium"]:
            return self._populate_fully_specific_prompt_lcb(sample)
        elif sample.get("source") == "humaneval":
            return self._populate_fully_specific_prompt_humaneval(sample)
        else:
            raise ValueError(f"Invalid source: {sample.get('source')}")

    def _populate_fully_specific_prompt_lcb(self, sample: Dict[str, Any]) -> str:
        query = sample["question_content"]
        formatting_preamble, starter_code = self.get_formatting_preamble(sample)

        return self.fully_specified_prompt_lcb.replace("[[QUESTION]]", query) \
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble) \
            .replace("[[FORMATTING]]", starter_code)

    def _populate_fully_specific_prompt_humaneval(self, sample: Dict[str, Any]) -> str:
        user_query = sample["prompt"]

        return self.fully_specified_prompt_humaneval.replace("[[INSTRUCTION]]", f"Complete the following incomplete function signature:\n```python\n{user_query}\n```")

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        if sample.get("source") in ["lcb_easy", "lcb_medium"]:
            return self._populate_concat_prompt_lcb(sample)
        elif sample.get("source") == "humaneval":
            return self._populate_concat_prompt_humaneval(sample)
        else:
            raise ValueError(f"Invalid source: {sample.get('source')}")

    def _populate_concat_prompt_lcb(self, sample: Dict[str, Any]) -> str:
        query = sample["shards"][0]["shard"] + "\n"
        for shard in sample["shards"][1:]:
            query += f"- {shard['shard']}\n"

        formatting_preamble, starter_code = self.get_formatting_preamble(sample)

        return self.fully_specified_prompt_lcb.replace("[[QUESTION]]", query) \
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble) \
            .replace("[[FORMATTING]]", starter_code)

    def _populate_concat_prompt_humaneval(self, sample: Dict[str, Any]) -> str:
        query = sample["shards"][0]["shard"] + "\n"
        for shard in sample["shards"][1:]:
            query += f"- {shard['shard']}\n"

        return self.fully_specified_prompt_humaneval.replace("[[INSTRUCTION]]", query)

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if sample.get("source") in ["lcb_easy", "lcb_medium"]:
            return {
                "task_id": sample["task_id"],
                "prompt": sample["question_content"] + "\n\n" + sample["starter_code"],
            }
        elif sample.get("source") == "humaneval":
            return {
                "task_id": sample["task_id"],
                "prompt": sample["prompt"],
            }

    def extract_answer(self, text: str) -> str:

        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)

        if "class Solution" in text:

            return [c for c in code_blocks if "class Solution" in c][-1]

        if code_blocks:
            for block in reversed(code_blocks):
                result = self._extract_function_from_code(block)
                if result:
                    return result
            return ""

        text = text.strip()
        if text.startswith("```") or text.startswith("`"):
            text = text[text.find("\n"):].strip()
        import_idx = text.rfind("import")
        def_idx = text.rfind("def")
        start_idx = import_idx if import_idx >= 0 else def_idx
        if start_idx >= 0:
            text = text[start_idx:]
        return self._extract_function_from_code(text)

    def _add_parent_info(self, node, parent=None):
        node.parent = parent
        for child in ast.iter_child_nodes(node):
            self._add_parent_info(child, node)

    def _extract_function_from_code(self, code: str) -> str:
        try:

            tree = ast.parse(code)

            self._add_parent_info(tree)

            import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            imports = []
            for node in import_nodes:
                imports.append(ast.unparse(node))

            function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not function_nodes:
                return ""

            last_function = None
            for node in reversed(function_nodes):

                if isinstance(node.parent, ast.Module):
                    last_function = node
                    break

            if not last_function:
                return ""

            source_lines = code.splitlines()

            start_line = (last_function.decorator_list[0].lineno - 1
                         if last_function.decorator_list
                         else last_function.lineno - 1)
            end_line = last_function.end_lineno

            function_text = '\n'.join(source_lines[start_line:end_line])

            if imports:
                return '\n'.join(imports) + '\n\n' + function_text
            return function_text
        except Exception:
            return ""

    def extract_function_body(self, code: str) -> str:
        try:

            tree = ast.parse(code)

            self._add_parent_info(tree)

            function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not function_nodes:
                print("No functions found")
                return ""

            last_function = None
            for node in reversed(function_nodes):

                if isinstance(node.parent, ast.Module):
                    last_function = node
                    break

            if not last_function:
                print("No top-level functions found")
                return ""

            source_lines = code.splitlines()

            start_line = last_function.lineno
            end_line = last_function.end_lineno

            body_lines = source_lines[start_line:end_line]

            min_indent = float('inf')
            for line in body_lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            body_lines = [line[min_indent:] if line.strip() else line for line in body_lines]

            return '\n'.join(body_lines)
        except Exception as e:
            print("Error extracting function body")
            print(e)
            return ""

if __name__ == "__main__":
    task = TaskCode()
    samples = task.get_samples()
    print(len(samples))
