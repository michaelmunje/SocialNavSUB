#!/usr/bin/env python3
"""
Gemini API Baseline Wrapper

Provides a clean interface to Google Gemini via the google.generativeai SDK,
with support for image encoding and conversation context.
"""
import os
import re
import time
import logging
import argparse
from typing import List, Tuple, Dict, Optional

import yaml
import google.generativeai as genai

from api_baseline import APIBaseline

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.
    Returns:
        Dictionary of configuration parameters.
    """
    with open(config_path, 'r') as fp:
        return yaml.safe_load(fp)


class Gemini(APIBaseline):
    """
    Wrapper around the Gemini API to generate text with optional images
    and past conversation context.
    """
    def __init__(
        self,
        model_name: str,
        api_key_env_var: Optional[str] = None
    ):
        """
        Initialize Gemini baseline with model name and API key environment variable.

        Args:
            model_name: Name of the Gemini model (e.g. 'gemini-2.0-flash-exp').
            api_key_env_var: Name of environment variable containing the API key.
        """
        super().__init__(model_name=model_name, api_key_env_var=api_key_env_var)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.logger = logging.getLogger(f"{__name__}.Gemini")

    def encode_image(self, image_path: str) -> str:
        """
        Read an image from disk and return its base64 encoding.

        Args:
            image_path: File system path to the image.
        Returns:
            Base64-encoded string of the image bytes.
        """
        with open(image_path, 'rb') as img_fp:
            return base64.b64encode(img_fp.read()).decode('utf-8')

    def generate_text_individual(
        self,
        text: str,
        image_filepaths: Optional[List[str]] = None
    ) -> str:
        """
        Generate a single response from Gemini given text and optional images.

        Args:
            text: Prompt text for the model.
            image_filepaths: List of base64-encoded image strings.
        Returns:
            JSON-like string extracted from model output.
        """
        return self._call_api(text, encoded_images=image_filepaths or [])

    def generate_text_using_past_conversations(
        self,
        text: str,
        image_filepaths: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using Gemini, including prior conversation history as context.

        Args:
            text: Current user prompt.
            image_filepaths: List of base64-encoded images.
        Returns:
            Extracted JSON-like string from response.
        """
        return self._call_api(
            text,
            encoded_images=image_filepaths or [],
            past_conversations=self.past_conversations
        )

    def _call_api(
        self,
        text: str,
        encoded_images: List[str],
        past_conversations: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Internal helper to structure and send a request to Gemini.

        Args:
            text: Prompt text.
            encoded_images: Base64-encoded image payloads.
            past_conversations: History of (user, assistant) pairs.
        Returns:
            Extracted JSON-like content from the model's raw response.
        """
        # Prepare image blocks
        blocks = []
        for img_b64 in encoded_images:
            blocks.append({'mime_type': 'image/jpeg', 'data': img_b64})

        # Combine with prompt text
        content_list = [*blocks, text]

        try:
            response = self.model.generate_content(content_list)
            raw_text = response.text

            # Extract JSON substring
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON-like content found in response.")
            return match.group()

        except Exception as e:
            self.logger.warning(
                "API call failed: %s. Retrying in 10s.", str(e)
            )
            time.sleep(10)
            return self._call_api(text, encoded_images, past_conversations)


def main():
    """
    CLI entrypoint: loads config, instantiates Gemini, and runs a demo call.
    """
    parser = argparse.ArgumentParser(
        description="Gemini API Baseline CLI"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='gemini_cfg.yaml',
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = Gemini(
        model_name=cfg['model_name'],
        api_key_env_var=cfg.get('api_key_env_var')
    )

    # Example usage
    prompt = cfg.get('demo_prompt', 'Hello, Gemini!')
    images = cfg.get('demo_images', [])
    result = client.generate_text_individual(prompt, images)
    logger.info("Demo result: %s", result)


if __name__ == '__main__':
    main()