from baseline import Baseline
from abc import ABC, abstractmethod
from typing import List
import os


class APIBaseline(Baseline):
    def __init__(self, 
                 model_name: str,
                 api_key_env_var: str):
        """
        Initialize the baseline model.
        This method should be implemented in each child class.
        """
        super().__init__(api_key_env_var)
        self.api_key = self.load_api_key(api_key_env_var)
        self.model_name = model_name
        self.baseline_type = 'api'

    def load_api_key(self, 
                     api_key_env_var: str):
        """
        Load the API key from an environment variable.

        :param api_key_env_var: The environment variable name where the API key is stored.
        :return: The API key as a string.
        :raises EnvironmentError: If the API key is not found in the environment.
        """
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise EnvironmentError(f"API key not found in environment variable: {api_key_env_var}")
        return api_key

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