#!/usr/bin/env python3
"""
LLaVa Baseline Wrapper

Implements a local Vision-Language baseline using Hugging Face LLaVa models,
with optional 4/8-bit quantization, chain-of-thought flag, and support for
both image and video inputs. Config-driven, with structured logging.
"""
import os
import re
import json
import logging
import argparse
from typing import List, Optional, Tuple, Dict

import torch
import numpy as np
import cv2
from PIL import Image
import yaml

from baseline import Baseline
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    BitsAndBytesConfig
)

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML config.
    Returns:
        Parsed configuration dict.
    """
    with open(path, 'r') as fp:
        return yaml.safe_load(fp)


class LLaVaBaseline(Baseline):
    """
    Local Vision-Language model wrapper for LLaVa.

    Supports optional quantization and chain-of-thought.
    """
    def __init__(
        self,
        model_name: str,
        use_cot: bool = False,
        quant_bits: Optional[int] = None
    ):
        """
        Initialize model, processor, and device.

        Args:
            model_name: HF identifier for the LLaVa model.
            use_cot: Enable chain-of-thought (not yet implemented).
            quant_bits: Load model in 4-bit or 8-bit if specified.
        """
        super().__init__(use_cot)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.quant_bits = quant_bits

        # Load quantization config if requested
        quant_cfg = None
        if quant_bits in (4, 8):
            kwargs = {
                'load_in_4bit': quant_bits == 4,
                'load_in_8bit': quant_bits == 8,
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_8bit_compute_dtype': torch.float16
            }
            quant_cfg = BitsAndBytesConfig(**kwargs)
            logger.info("Quantization enabled: %d-bit", quant_bits)

        # Decide on video vs image model
        if 'video' in model_name.lower():
            self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_cfg
            )
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_cfg
            )
            # add special tokens
            self.processor.tokenizer.add_tokens(['<image>', '<pad>'], special_tokens=True)
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.model.to(self.device)
        self.baseline_type = 'local'
        logger.info("Loaded model '%s' on %s", model_name, self.device)

    def extract_json(self, text: str) -> str:
        """
        Extract JSON-like substring from model output.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON content found in model output.")
        return match.group()

    def generate_text_individual(
        self,
        prompt: str,
        images: List[str]
    ) -> str:
        """
        Single-turn generation with image(s) or video frames.

        Args:
            prompt: Textual question.
            images: List of image file paths or numpy arrays.
        Returns:
            JSON-like string from model.
        """
        # Image mode
        if 'video' not in self.model_name.lower():
            assert len(images) == 1, "Only one image supported"
            img = images[0]
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            else:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            conv = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': prompt}
            ]}]
            chat = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = self.processor(text=chat, images=img, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            ids = self.model.generate(**inputs, max_new_tokens=512)
            out = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

        else:
            # Video mode
            assert len(images) > 1, "Provide multiple frames for video"
            frames = []
            for frame in images:
                arr = frame if isinstance(frame, np.ndarray) else np.array(Image.open(frame))
                frames.append(arr)
            clip = np.stack(frames)

            conv = [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt}, {'type': 'video'}
            ]}]
            chat = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = self.processor(text=chat, videos=clip, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            ids = self.model.generate(**inputs, max_new_tokens=512)
            out = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

        # Extract JSON and retry if necessary
        try:
            return self.extract_json(out)
        except ValueError:
            logger.warning("Output not JSON, retrying...")
            return self.generate_text_individual(prompt, images)

    def generate_text_using_past_conversations(
        self,
        prompt: str,
        images: List[str]
    ) -> str:
        """
        Chain-of-thought generation (not implemented).
        """
        raise NotImplementedError("CoT generation is not yet supported.")


def main():
    """
    CLI entrypoint: load config, instantiate baseline, run demo.
    """
    parser = argparse.ArgumentParser(description="LLaVa Baseline CLI")
    parser.add_argument(
        '--config', type=str, default='llava_cfg.yaml',
        help='Path to YAML configuration'
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    client = LLaVaBaseline(
        model_name=cfg['model_name'],
        use_cot=cfg.get('use_cot', False),
        quant_bits=cfg.get('quantization_bits')
    )

    # Demo
    demo_text = cfg.get('demo_prompt', 'Describe the scene')
    demo_imgs = cfg.get('demo_images', [])
    result = client.generate_text_individual(demo_text, demo_imgs)
    logger.info("Demo output: %s", result)


if __name__ == '__main__':
    main()