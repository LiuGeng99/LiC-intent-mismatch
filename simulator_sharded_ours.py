import json
import random

from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from system_agent import SystemAgent
from model_openai import generate
from user_agent import UserAgent
from ours_agent import Agent_A
from tasks import get_task

from mem0 import Memory
import os
from pathlib import Path

def qwen_generate(*args, **kwargs):
    raise RuntimeError("Local Qwen generation is not part of this paper snapshot.")

class ConversationSimulatorSharded:
    def __init__(self, sample, assistant_model="gpt-4o-mini", system_model="gpt-4o-mini", user_model="gpt-4o-mini", assistant_temperature=1.0, user_temperature=1.0, dataset_fn=None, log_folder="logs"):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.system_model = system_model
        self.user_model = user_model
        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(self.task_name, system_model, self.sample)
        self.log_folder = log_folder
        self.system_message = self.task.generate_system_prompt(self.sample)
        self.answer_description = self.task.get_answer_description()

        self.run_with_custom_temperature = assistant_temperature != 1.0 or user_temperature != 1.0
        self.assistant_temperature = assistant_temperature
        self.user_temperature = user_temperature

        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        self.user_id = "debug"
        
        self.task_id = sample["task_id"]
        self.identifier = f"{self.task_name}_{self.task_id}"
        

        self.agent_a = Agent_A(self.task, model="gpt-4o")
        self.infer_trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]

        self.memory_name = "mem0"
        if self.memory_name == "mem0":
            config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.2,
                        "max_tokens": 2000,
                    }
                },
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "test",
                        "path": str(Path(__file__).resolve().parent / "logs" / "mem0_chroma" / f"db_{self.identifier}"),
                    }
                }
            }
            self.memory_client = Memory.from_config(config)

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def memory_ours(self, input):
        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000

        assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
        assistant_response = assistant_response_obj["message"]
            
        log_path = f"/slurm/geng_liu/Intent_paper/reproduce/logs/conversation_ours/{self.identifier}"
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        
        conversation_path = f"{log_path}/conversation.jsonl"
        memory_path = f"{log_path}/memory.jsonl"
        
        total_usd = 0.0
        if self.iteration == 0:
            history_iter = "None"
        else:
            with open(memory_path, "r", encoding="utf-8") as file:
                for i, line in enumerate(file, 1):
                    memory_iter = json.loads(line)
                    history_iter = memory_iter["history"]

            

        summary_message = self.memory_client.summary(history_iter)
        context_for_key_info = self.trace + summary_message
        current_key_info_obj = generate(extract_conversation(context_for_key_info, to_str=False), \
            model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
        current_key_info = current_key_info_obj["message"]
        

        trace_with_response = self.trace + [{"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]}]
        context_for_refine = trace_with_response + self.memory_client.refine(current_key_info)
        current_refine_obj = generate(extract_conversation(context_for_refine, to_str=False), \
            model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
        current_refine = current_refine_obj["message"]
        
         
        
        memory_json = {"iter":self.iteration, "input":input, "history":current_key_info, "refine":current_refine}
        with open(memory_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(memory_json)+"\n")
           

        with open(conversation_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(f"[User]: {input}\n[AI]: {current_refine}")+"\n")
        
        total_usd = 0
        return {"message":current_refine, "total_usd":total_usd}
    
    
    def run_ours(self, verbose=False, save_log=True):
        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        self.iteration = -1
        while not is_completed:
            self.iteration += 1
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            assistant_response_obj = self.memory_ours(user_response)

            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)

            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":

                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:

                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score
    
    def run_mem0(self, verbose=False, save_log=True):
        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            
            

            relevant_memories = self.memory_client.search(user_response, user_id=self.user_id)

            memories_str = "\\n".join(f"- {memory['memory']}" for memory in relevant_memories["results"])
            self.trace[-1]["content"] = f"{user_response} \\n Here are some key information: \\n {memories_str}"

                
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            if "Qwen" in self.assistant_model:
                assistant_response_obj = qwen_generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            else:
                assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            

            

            self.memory_client.add([{"role": "user", "content": user_response}], user_id="debug")

            
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)

            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":

                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:

                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score
    
    def run_memos(self, verbose=False, save_log=True):
        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            
            

            relevant_memories = self.memory_client.search(user_response, user_id=self.user_id)

            memories_str = "\\n".join(f"- {memory['memory']}" for memory in relevant_memories["results"])

            

            self.trace[0]["content"] = f"{self.system_message} \\n Ensure your response satisfies all key information in following memories. \\n Memories: \\n {memories_str}"
               
                
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            if "Qwen" in self.assistant_model:
                assistant_response_obj = qwen_generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            else:
                assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            

            self.memory_client.add([{"role": "user", "content": user_response},\
                {"role": "assistant", "content": assistant_response}], user_id="debug")

            
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)

            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":

                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:

                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score

    def run_modified(self, verbose=False, save_log=True):

        
        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

                    

            if "Qwen" in self.assistant_model:
                assistant_response_obj = qwen_generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            else:
                assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]

            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
            

            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":

                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:

                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score
    

    
    def run(self, verbose=False, save_log=True):

        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            infer_response, infer_cost_usd = self.agent_a.generate_response(self.trace, self.sample, temperature=self.user_temperature)

            self.trace.append({"role": "infer", "content": infer_response, "timestamp": date_str(), "cost_usd": infer_cost_usd})
            

            assistant_response_obj = generate(extract_conversation(self.infer_trace+[{"role": "user", "content": infer_response, "timestamp": date_str(), "cost_usd": infer_cost_usd}], to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})

            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)

            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":

                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:

                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    dataset_fn = "data/sharded_instructions_600.json"
    with open(dataset_fn, "r") as f:
        data = json.load(f)

    data = [d for d in data if d["task"] == args.task]

    sample = random.choice(data)

    conversation_simulator = ConversationSimulatorSharded(sample=sample, assistant_model=args.assistant_model, system_model=args.system_model, user_model=args.user_model, dataset_fn=dataset_fn)
    conversation_simulator.run(verbose=True, save_log=True)
    
    
