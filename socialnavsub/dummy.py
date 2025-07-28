import json
import re
from typing import List, Tuple
from baseline import Baseline

class DummyBaseline(Baseline):
    def __init__(self):
        """
        Initialize the baseline model.
        This method should be implemented in each child class.
        """
        super().__init__()
        self.baseline_type = 'api'
    
    def generate_text_individual(self, 
                                 text: str, 
                                 image_filepaths: List[str] = []):
        """
        Generate text by selecting the first possible answer from the provided text.
        
        :param text: The input text containing the question and possible answers.
        :param image_filepaths: Optional image paths (not used in this dummy implementation).
        :return: A JSON string with the selected answer.
        """
        possible_answers = self.extract_possible_answers(text)
        assert len(possible_answers) > 0, f"Invalid possible answers {possible_answers}"
        first_answer = possible_answers[0]
        return json.dumps({"answer": first_answer})

    def generate_text_using_past_conversations(self,
                                               text: str,
                                               image_filepaths: List[str] = []):
        """
        Generate text using past conversations (dummy implementation).
        
        :param text: The input text containing the question and possible answers.
        :param image_filepaths: Optional image paths (not used in this dummy implementation).
        :return: A JSON string with the selected answer.
        """
        return self.generate_text_individual(text, image_filepaths)
    
    def extract_possible_answers(self, text: str) -> List[str]:
        """
        Extract possible answers from the input text.
        
        :param text: The input text containing possible answers.
        :return: A list of possible answers.
        """
        # Find the line containing "Possible answers:"
        matches = re.findall(r'Possible answers:\s*(.*)', text)
        if matches:
            # Extract answers enclosed in double quotes
            answers = re.findall(r'"([^"]+)"', matches[0])
            return answers
        return []
