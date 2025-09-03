#!/usr/bin/env python3
"""
Evaluation script for baseline VLM models.

Loads survey data, prompts the model, computes metrics, and saves structured
results for each sample.
"""
import os
import json
import base64
import argparse
import logging
from typing import Dict
from time import perf_counter 

import cv2
import tqdm

import numpy as np
import shutil

from utils import compute_top_k_accuracy, compute_top_k_human_accuracy, compute_cross_entropy, Answer

from survey_loader import (
    load_survey_questions_independent,
    load_survey_questions_cot,
    load_survey_questions_cot_with_gt,
    get_image_prompt,
)
from utils import (
    load_yaml,
    load_model_class,
    load_human_answer,
    copy_config_files,
    validate_prompts_in_human_answers,
    save_debug_images,
    REASONING_GROUPS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def evaluate_baseline(
    baseline_model: str,
    survey_folder: str,
    prompts_folder: str,
    evaluation_folder: str,
    model_to_api_key: Dict,
    prompt_image_fp: str,
    method: str = "independent",
    prompts_data: str = "prompts/survey_prompt.json",
    samples_json_fp: str = "prompts/sample_info.json",
    relevant_prev_qs_fp: str = "prompts/relevant_prev_questions.json",
    config: Dict = None,
    debug: bool = False,
    resume_folder: str = None,
) -> None:
    """
    Evaluate a baseline model on human survey data and save metrics.

    Parameters:
        baseline_model: Name of the model class to load.
        survey_folder: Directory containing human survey answer folders.
        prompts_folder: Directory containing prompt JSON files.
        evaluation_folder: Base directory to store evaluation outputs.
        model_to_api_key: Mapping from model names to API keys (for API-based models).
        prompt_image_fp: File path template for image prompts.
        method: One of 'independent', 'cot', 'cot_with_gt'.
        prompts_data: Path to survey_prompt.json.
        samples_json_fp: Path to sample_info.json.
        relevant_prev_qs_fp: Path to relevant_prev_questions.json.
        config: Additional configuration dict (e.g., from config.yaml).
        dataset_cfg: Path to dataset configuration YAML.
        debug: If True, save debug images.
        resume_folder: If provided, resume from an existing eval folder.
    """
    logger.info("Initializing evaluation: model=%s, method=%s", baseline_model, method)
    model = load_model_class(baseline_model, model_to_api_key)
    assert method in ("independent", "cot", "cot_with_gt"), "Invalid method"
    
    dataset_cfg = config['dataset_cfg_fp']

    # Load sample definitions
    with open(samples_json_fp, 'r') as fp:
        samples = json.load(fp)
    sample_ids = [s['folderpath'] for s in samples['samples']]
    sample_dirs = [os.path.join(prompts_folder, sid) for sid in sample_ids]
    answer_dirs = [os.path.join(survey_folder, sid) for sid in sample_ids]
    n_people_list = [s['n_people'] for s in samples['samples']]

    # Verify paths
    for path in sample_dirs + answer_dirs:
        if not os.path.isdir(path):
            logger.error("Required directory not found: %s", path)
            raise FileNotFoundError(path)

    # Setup evaluation output directory
    if resume_folder:
        logger.info("Resuming from: %s", resume_folder)
        processed = set(os.listdir(resume_folder))
        sample_dirs = [d for d in sample_dirs if os.path.basename(d) not in processed]
        eval_base = resume_folder
    else:
        if not os.path.exists(evaluation_folder):
            os.makedirs(evaluation_folder, exist_ok=True)
        existing = [d for d in os.listdir(evaluation_folder)
                    if os.path.isdir(os.path.join(evaluation_folder, d))]
        idx = len(existing) + 1
        eval_base = os.path.join(
            evaluation_folder,
            f"experiment_{idx}_{baseline_model}_{method}"
        )
        logger.info("Creating evaluation directory: %s", eval_base)
        os.makedirs(eval_base, exist_ok=True)

    # Load previous-question map
    with open(relevant_prev_qs_fp, 'r') as fp:
        relevant_prev_qs = json.load(fp)

    # Copy config for reproducibility
    if not resume_folder:
        copy_config_files(
            eval_base,
            prompts_data,
            samples_json_fp,
            relevant_prev_qs_fp,
            dataset_cfg
        )

    # Load dataset config if needed
    _ = load_yaml(dataset_cfg)

    # Iterate samples
    for idx, (s_dir, a_dir, n_ppl) in enumerate(
        zip(sample_dirs, answer_dirs, n_people_list), start=1
    ):
        # Choose prompting method
        if method == 'independent':
            prompts = load_survey_questions_independent(prompts_data, n_ppl)
        elif method == 'cot':
            prompts = load_survey_questions_cot(
                prompts_data, n_ppl, relevant_prev_qs
            )
        else:
            prompts = load_survey_questions_cot_with_gt(
                prompts_data, n_ppl, relevant_prev_qs
            )

        validate_prompts_in_human_answers(a_dir, prompts)

        sample_id = os.path.basename(s_dir)
        logger.info("Processing sample %d/%d: %s", idx, len(sample_dirs), sample_id)
        out_dir = os.path.join(eval_base, sample_id)
        os.makedirs(out_dir, exist_ok=True)

        # Prepare image prompt
        parent_dir = os.path.dirname(prompt_image_fp)
        parts = sample_id.split('_')
        image_fp = os.path.join(
            parent_dir,
            f"{parts[0]}_{parts[1]}_{parts[2]}",
            f"{parts[3]}.jpg"
        )
        images = get_image_prompt(dataset_cfg, image_fp, config)
        if model.baseline_type == 'api':
            encoded = []
            for img in images:
                resized = cv2.resize(img, (512, 512))
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                data = cv2.imencode('.jpg', bgr)[1]
                encoded.append(base64.b64encode(data).decode())
            images = encoded

        if debug:
            save_debug_images(baseline_model, model, images, sample_id)

        # Evaluate prompts
        human_fp = os.path.join(a_dir, 'common_answers.json')
        metrics = {}
        history = []
        # --- per-sample accumulators ---
        question_answers_export = {}
        question_answers = {}
        human_question_answers = {}
        eval_export = {}
        n_vlm_queries = 1  # single-shot per question

        for q_idx, (q_key, prompt, choices, q_type) in enumerate(tqdm.tqdm(prompts)):
            # Determine reasoning group
            base_q = q_key if q_key.startswith('q_goal_position') else q_key.rsplit('_p', 1)[0]
            group = next(
                (g for g, qs in REASONING_GROUPS.items() if base_q in qs),
                None
            )
            if group is None:
                raise ValueError(f"Unknown question group for {base_q}")

            # Get model or GT answer
            if method == 'cot_with_gt' and group in (
                'Spatial reasoning', 'Spatiotemporal reasoning'
            ):
                human_ans = load_human_answer(human_fp, q_key, choices, q_type)
                clean_answer = human_ans.get_most_common_answer()
            else:
                t0 = perf_counter()
                # print logger
                ans_raw = model.generate_text(prompt, images)
                latency_s = perf_counter() - t0
                history.extend([
                    {'entity': ['user'], 'response': prompt.split('\n')},
                    {'entity': ['assistant'], 'response': [ans_raw], 'latency_sec': latency_s}
                ])
                with open(os.path.join(out_dir, 'conversation.json'), 'w') as cfp:
                    json.dump(history, cfp, indent=4)
                try:
                    clean_answer = json.loads(ans_raw)['answer']
                except Exception:
                    clean_answer = 'INVALID'
                    
            # ---- Save / compute metrics & update CoT prompts ----
            question_answers_export[q_key] = clean_answer

            if n_vlm_queries == 1:
                # one-hot prediction with prob=1.0 on the chosen answer
                question_answers[q_key] = Answer(
                    [clean_answer],
                    [1.0],
                    choices,
                    len(choices),
                    q_type
                )
            else:
                raise NotImplementedError('Multiple VLM queries not implemented')

            # If using CoT, inject this answer into later prompts and (optionally) the model history
            if 'cot' in method:
                for j in range(q_idx + 1, len(prompts)):
                    fqkey, fprompt, fchoices, fqtype = prompts[j]
                    answer_dummy_txt = "{QUESTION_KEY_ANSWER}".replace("QUESTION_KEY", q_key)
                    if isinstance(clean_answer, str):
                        clean_answer_with_dict = "{\"answer\": " + clean_answer + "}"
                    else:
                        clean_answer_with_dict = "{\"answer\": \"" + str(clean_answer) + "\"}"
                    fprompt = fprompt.replace(answer_dummy_txt, clean_answer_with_dict)
                    prompts[j] = (fqkey, fprompt, fchoices, fqtype)

                # best-effort: keep conversational state if the model supports it
                try:
                    model.add_to_conversation_history(('user', prompt))
                    model.add_to_conversation_history(('assistant', json.dumps({"answer": clean_answer})))
                except Exception:
                    pass

            # Load human distribution for this question
            human_question_answers[q_key] = load_human_answer(human_fp, q_key, choices, q_type)

            # Metrics
            top_1_accuracy = compute_top_k_accuracy(question_answers[q_key], human_question_answers[q_key], 1)
            top_2_accuracy = compute_top_k_accuracy(question_answers[q_key], human_question_answers[q_key], 2)
            cross_entropy = compute_cross_entropy(question_answers[q_key], human_question_answers[q_key])

            vlm_entropy = -np.sum([p * np.log2(p) for p in question_answers[q_key].answers_probabilities if p > 0])
            human_entropy = -np.sum([p * np.log2(p) for p in human_question_answers[q_key].answers_probabilities if p > 0])

            top_1_human_accuracy = compute_top_k_human_accuracy(human_question_answers[q_key], 1)
            top_2_human_accuracy = compute_top_k_human_accuracy(human_question_answers[q_key], 2)

            top_1_random_accuracy = 1 / len(human_question_answers[q_key].choices)
            top_2_random_accuracy = 2 / len(human_question_answers[q_key].choices)

            # Export structure for this question
            eval_export[q_key] = {
                'top_1_accuracy': top_1_accuracy,
                'top_2_accuracy': top_2_accuracy,
                'top_1_human_accuracy': top_1_human_accuracy,
                'top_2_human_accuracy': top_2_human_accuracy,
                'top_1_random_accuracy': top_1_random_accuracy,
                'top_2_random_accuracy': top_2_random_accuracy,
                'cross_entropy': cross_entropy,
                'vlm_entropy': float(vlm_entropy),
                'human_entropy': float(human_entropy),
                'latency_sec': latency_s if 'latency_s' in locals() else None,
            }

            # Add probability maps (cast keys to str for JSON)
            eval_export[q_key]['vlm_probabilities'] = {
                str(k): v for k, v in question_answers[q_key].answer_to_probability.items()
            }
            eval_export[q_key]['human_probabilities'] = {
                str(k): v for k, v in human_question_answers[q_key].answer_to_probability.items()
            }

            # Persist progress each question
            with open(os.path.join(out_dir, 'question_answers.json'), 'w') as of:
                json.dump(question_answers_export, of, indent=4)

            # copy the human common answers once per sample (cheap to overwrite)
            shutil.copy(human_fp, os.path.join(out_dir, 'human_common_answers.json'))

            with open(os.path.join(out_dir, 'evaluation.json'), 'w') as of:
                json.dump(eval_export, of, indent=4)

        logger.info("Finished sample: %s", sample_id)

    logger.info("All samples processed.")


def main() -> None:
    """
    Entry point for the evaluation script.
    Parses CLI arguments, loads configuration, and runs evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate baseline VLM model against human survey data."
    )
    parser.add_argument(
        '--resume_folder',
        type=str,
        default=None,
        help='Folder to resume evaluation from'
    )
    parser.add_argument(
        '--cfg_path',
        type=str,
        default='config.yaml',
        help='Path to the config file'
    )
    args = parser.parse_args()

    if args.resume_folder:
        logger.info("Resuming evaluation from %s", args.resume_folder)

    cfg = load_yaml(args.cfg_path)
    evaluate_baseline(
        baseline_model=cfg['baseline_model'],
        survey_folder=cfg['survey_folder'],
        prompts_folder=cfg['prompts_folder'],
        evaluation_folder=cfg['evaluation_folder'],
        model_to_api_key=cfg['model_to_api_key'],
        prompt_image_fp=cfg['prompt_image_fp'],
        method=cfg['method'],
        prompts_data=os.path.join(cfg['prompts_folder'], 'survey_prompt.json'),
        samples_json_fp=os.path.join(cfg['prompts_folder'], 'sample_info.json'),
        relevant_prev_qs_fp=os.path.join(
            cfg['prompts_folder'], 'relevant_prev_questions.json'
        ),
        config=cfg,
        debug=cfg.get('debug', False),
        resume_folder=args.resume_folder,
    )


if __name__ == '__main__':
    main()
