from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import json

class Task(ABC):

    def __init__(self, version: str):
        self.version: str = version
        self.task_name: str = self._get_task_name()
        self.answer_description: str = self._get_answer_description()
        self.answer_extraction_strategy: str = self._get_answer_extraction_strategy()

    @abstractmethod
    def get_task_name(self) -> str:
        pass

    @abstractmethod
    def get_dataset_file(self) -> str:
        pass

    @abstractmethod
    def get_samples(self, version: str, filter: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        samples = self.get_samples()
        id2samples = {sample["task_id"]: sample for sample in samples}
        if sample_id not in id2samples:
            raise ValueError(f"Sample ID {sample_id} not found")
        return id2samples[sample_id]

    @abstractmethod
    def get_answer_description(self) -> str:
        pass

    @abstractmethod
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> Union[bool, tuple[bool, str]]:
        pass

    @abstractmethod
    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        pass

    def save_samples(self, samples: List[Dict[str, Any]]):
        dataset_fn = self.get_dataset_file()
        with open(dataset_fn, "w") as f:
            json.dump(samples, f, indent=4)

    @abstractmethod
    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass
