from abc import ABC, abstractmethod
from typing import List, Tuple

class Baseline(ABC):
    def __init__(self, use_cot: bool = False):
        """
        Initialize the baseline model.
        This method should be implemented in each child class.
        """
        super().__init__()
        self.use_cot: bool = use_cot
        # past conversation example: [('user', 'Hello!'), ('assistant', 'Hi!')]
        self.past_conversations: List[Tuple[str, str]] = []
        self.baseline_type = None
        
    def generate_text(self, 
                      text: str, 
                      image_filepaths: List[str] = []):
        """
        Generate text based on the provided text and optionally an image.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        if self.use_cot:
            return self.generate_text_using_past_conversations(text, image_filepaths)
        else:
            return self.generate_text_individual(text, image_filepaths)

    @abstractmethod
    def generate_text_individual(self, 
                                 text: str, 
                                 image_filepaths: List[str] = []):
        """
        Generate text based on the provided text and images.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        pass
    
    @abstractmethod
    def generate_text_using_past_conversations(self,
                                               text: str,
                                               image_filepaths: List[str] = []):
        """
        Generate text based on the provided text, images, and past conversations.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        pass
    
    def add_to_conversation_history(self, conversation: Tuple[str, str]):
        """
        Set the past text to be used in generating the next response.
        This is primarily used when using chain-of-thought with ground truth.

        :param text: The past text to set.
        """
        self.past_conversations += [conversation]
        
    def clear_conversation_history(self):
        """
        Clear the conversation history.
        """
        self.past_conversations = []
