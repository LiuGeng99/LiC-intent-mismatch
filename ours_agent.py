import json
from pathlib import Path

from model_openai import generate_json, generate
from utils import extract_conversation
from mediator import Mediator

import re

ROOT = Path(__file__).resolve().parent

Agent_A = Mediator

class Agent_sum:
    def __init__(self, task, model="gpt-5.1"):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()

        with open(ROOT / "prompts/mediator.txt", "r") as f:
            self.prompt_response = f.read()

    def generate_response(self, conversation, sample, temperature=1.0):   
        conversation_message = ""
        for msg in conversation:
            if msg["role"] == "user":
                conversation_message += f"[User]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_message += f"[AI]: {msg['content']}\n"
                
        agent_a_message = self.prompt_response.replace("[[CONVERSATION]]", conversation_message)
        response = generate([{"role": "user", "content": agent_a_message}], model=self.model, temperature=temperature, return_metadata=True)
        
        return response["message"], response["total_usd"]
    
    
class Agent_B:
    def __init__(self, task, model="gpt-5.1"):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()
        with open(ROOT / "prompts/mediator.txt", "r") as f:
            self.prompt_response = f.read()

    def generate_response(self, conversation, sample, temperature=1.0): 
        conversation_message = ""
        for msg in conversation:
            if msg["role"] == "user":
                conversation_message += f"[User]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_message += f"[AI]: {msg['content']}\n"
                
        agent_a_message = self.prompt_response.replace("[[CONVERSATION]]", conversation_message)
        response = generate([{"role": "user", "content": agent_a_message}], model=self.model, temperature=temperature, return_metadata=True)
        
        return response["message"], response["total_usd"]
    
class Agent_C:
    def __init__(self, task, model="gpt-5.1"):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()
        with open("test_code/personalized_agent_c_prompt.txt", "r") as f:
            self.prompt_response = f.read()

    def generate_response(self, conversation): 
        conversation_message = ""
        for msg in conversation:
            if msg["role"] == "user":
                conversation_message += f"[User]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_message += f"[AI]: {msg['content']}\n"
                
        agent_c_message = self.prompt_response.replace("[[CONVERSATION]]", conversation_message)

        
        return agent_c_message
