import os
import yaml
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import argparse
from utils import (
    REASONING_GROUPS,
    find_evaluation_files,
    process_evaluation_file,
    compute_average,
    compute_kl_divergence,
    compute_metrics,
    write_eval_full_csv,
    write_eval_disagreement_csv,
    write_eval_aggregated_vlm_csv,
    write_eval_aggregated_human_csv,
    write_vlm_probabilities_csv,
    write_confusion_matrices_txt
)

class CumulativeMetric:
    def __init__(self, name):
        self.name = name
        self.values = []
        
    # include sample
    def include(self, value):
        self.values.append(value)
    
    def get_n_samples(self):
        return len(self.values)
    
    def get_average(self):
        return sum(self.values) / len(self.values)
    
    def get_std_error(self):
        if len(self.values) <= 1:
            return 0.0
        return np.std(self.values) / math.sqrt(len(self.values))


def compute_averages_and_generate_csv(root_dir, survey_folder, entropy_threshold=0.5):
    # Step 1: Find all evaluation files grouped by experiment folder
    experiment_files = find_evaluation_files(root_dir)

    all_models_full_experiment_results = []

    all_results = []
    disagreement_results = []
    aggregated_results_vlm = []
    aggregated_results_human = []

    # Dictionaries to collect probabilities across all experiments
    base_question_vlm_probs = {}
    base_question_human_probs = {}
    base_question_possible_answers = {}
    base_question_reasoning_group = {}

    # Overall confusion matrices for "All" category
    overall_confusion_matrix_vlm = defaultdict(int)
    overall_confusion_matrix_human = defaultdict(int)

    # Step 2: Process each experiment folder (model)
    for experiment_folder, evaluation_files in experiment_files.items():
        print("Processing experiment:", experiment_folder)
        experiment_path = os.path.join(root_dir, experiment_folder)
        answers_affected = {} # person -> answer
        # Read config.yaml inside the experiment folder
        eval_cfg_path = os.path.join(experiment_path, 'config.yaml')
        if os.path.isfile(eval_cfg_path):
            with open(eval_cfg_path, 'r') as file:
                eval_config = yaml.safe_load(file)
            baseline_model = eval_config.get('baseline_model', '')
            method = eval_config.get('method', '')
            prompt_image_type = eval_config.get('prompt_img_type', '')
        else:
            baseline_model = ''
            method = ''
            prompt_image_type = ''

        full_experiment_results = []
        experiment_results = []
        experiment_disagreement_results = []
        base_question_data = defaultdict(list)

        confusion_matrix_vlm = defaultdict(lambda: defaultdict(int))
        confusion_matrix_human = defaultdict(lambda: defaultdict(int))
        
        # make some metrics for this particular experiment
            
        # map base_question to CumulativeMetric
        question_cumulative_vlm_prob_agreement = {}
        question_cumulative_human_prob_agreement = {}
        question_cumulative_vlm_weighted_prob_agreement = {}
        question_cumulative_human_weighted_prob_agreement = {}
        question_cumulative_human_oracle_prob_agreement = {}
        question_cumulative_human_oracle_weighted_prob_agreement = {}

        # map base_question's reasoning group to CumulativeMetric
        reasoning_group_cumulative_vlm_prob_agreement = {}
        reasoning_group_cumulative_human_prob_agreement = {}
        reasoning_group_cumulative_vlm_weighted_prob_agreement = {}
        reasoning_group_cumulative_human_weighted_prob_agreement = {}
        reasoning_group_cumulative_human_oracle_prob_agreement = {}
        reasoning_group_cumulative_human_oracle_weighted_prob_agreement = {}
        
        # across all questions
        all_vlm_prob_agreement = CumulativeMetric("vlm_prob_agreement")
        all_human_prob_agreement = CumulativeMetric("human_prob_agreement")
        all_vlm_weighted_prob_agreement = CumulativeMetric("vlm_weighted_prob_agreement")
        all_human_weighted_prob_agreement = CumulativeMetric("human_weighted_prob_agreement")
        all_human_oracle_prob_agreement = CumulativeMetric("human_oracle_prob_agreement")
        all_human_oracle_weighted_prob_agreement = CumulativeMetric("human_oracle_weighted_prob_agreement")

        # Step 3: Process each evaluation.json file (per sample) within the experiment
        for file_path in evaluation_files:
            sample_folder = os.path.basename(os.path.dirname(file_path))
            if sample_folder == 'better_heuristic_results':
                continue # Skip this folder
            data = process_evaluation_file(file_path)
            human_answers_file = os.path.join(survey_folder, sample_folder, 'common_answers.json')
            assert os.path.isfile(human_answers_file), f"File not found: {human_answers_file}"
            with open(human_answers_file, 'r') as file:
                human_answers_data = yaml.safe_load(file)
            for question, evaluation in data.items():
                
                # make sure question is in the human answers
                if question not in human_answers_data or question + '_n_answers' not in human_answers_data:
                    print(f"Could not find question {question} in human answers for sample {sample_folder}.")
                    continue
                
                top_1_accuracy = evaluation["top_1_accuracy"]
                top_2_accuracy = evaluation.get("top_2_accuracy", 0)
                # human_probabilities = evaluation.get("human_probabilities", {})
                human_entropy = evaluation.get("human_entropy", 0)
                vlm_probabilities = evaluation.get("vlm_probabilities", {})
                
                # for our logging, check if 
                m_affected = re.match(r'q_robot_affected_p(\d+)', question)
                m_action = re.match(r'q_robot_action_p(\d+)', question)
                if m_affected:
                    p_id = m_affected.group(1)
                    # get most common answer
                    chosen_answer = max(vlm_probabilities, key=vlm_probabilities.get)
                    answers_affected[p_id] = chosen_answer
                if m_action:
                    p_id = m_action.group(1)
                    if p_id in answers_affected:
                        chosen_answer = max(vlm_probabilities, key=vlm_probabilities.get)
            for question, evaluation in data.items():
                if question not in human_answers_data or question + '_n_answers' not in human_answers_data:
                    print(f"Could not find question {question} in human answers for sample {sample_folder}.")
                    continue
                
                top_1_accuracy = evaluation["top_1_accuracy"]
                top_2_accuracy = evaluation.get("top_2_accuracy", 0)
                # human_probabilities = evaluation.get("human_probabilities", {})
                human_entropy = evaluation.get("human_entropy", 0)
                vlm_probabilities = evaluation.get("vlm_probabilities", {})
                
                # for our logging, check if 
                m_affected = re.match(r'q_robot_affected_p(\d+)', question)
                m_action = re.match(r'q_robot_action_p(\d+)', question)
                if m_affected:
                    p_id = m_affected.group(1)
                    # get most common answer
                    chosen_answer = max(vlm_probabilities, key=vlm_probabilities.get)
                    answers_affected[p_id] = chosen_answer
                if m_action:
                    p_id = m_action.group(1)
                    if p_id in answers_affected:
                        chosen_answer = max(vlm_probabilities, key=vlm_probabilities.get)

                human_probabilities = {choice: 0.0 for choice in vlm_probabilities.keys()}
                # if we can't find the human answers, print it out
                if question + '_n_answers' not in human_answers_data:
                    print(f"Sample {sample_folder} has no human answers for question {question}.")
                    print(f"Skipping sample {sample_folder} for question {question}.")
                    continue
                n_humans = human_answers_data[question + '_n_answers']
                if n_humans <= 1:
                    print(f"Skipping sample {sample_folder} for question {question} due to insufficient human answers.")
                    continue
                selected_human_choices = human_answers_data[question]
                selected_human_probabilities = human_answers_data[question + '_probabilities']
                for choice, prob in zip(selected_human_choices, selected_human_probabilities):
                    if type(choice) == list:
                        choice = str(tuple(choice))
                    assert choice in vlm_probabilities, f"Choice {choice} not in VLM probabilities for question {question}"
                    human_probabilities[choice] = prob
                kl_divergence = compute_kl_divergence(human_probabilities, vlm_probabilities)
                top_1_random_accuracy = evaluation.get("top_1_random_accuracy", 0)
                top_2_random_accuracy = evaluation.get("top_2_random_accuracy", 0)

                # Compute Top 1 and Top 2 Human Accuracy
                if human_probabilities:
                    
                    sorted_probs = sorted(human_probabilities.values(), reverse=True)
                    max_human_probability = sorted_probs[0]
                    top_1_human_accuracy = sorted_probs[0]
                    if len(sorted_probs) > 1:
                        top_2_human_accuracy = sorted_probs[1] + top_1_human_accuracy
                    else:
                        top_2_human_accuracy = top_1_human_accuracy
                else:
                    max_human_probability = 1e-9  # Avoid division by zero
                    top_1_human_accuracy = 0
                    top_2_human_accuracy = 0

                assert top_2_accuracy >= top_1_accuracy
                assert top_2_human_accuracy >= top_1_human_accuracy
                # Compute Probability of Agreement for VLM
                all_answers = set(human_probabilities.keys()).union(set(vlm_probabilities.keys()))
                # If a probability doesn't exist for an answer, default to 0
                vlm_prob_agreement = sum(human_probabilities.get(a, 0) * vlm_probabilities.get(a, 0) for a in all_answers)


                human_prob_contribution = 1.0 / n_humans
                n_pas_included = 0

                # counts for each human answer
                human_answer_counts = {a: int(p * n_humans) for a, p in human_probabilities.items()}
                human_labels = []
                for ans, count in human_answer_counts.items():
                    for _ in range(count):
                        human_labels.append(ans)
                
                # for each count, compute the probability of agreement and holdout the answer the current human said
                human_prob_agreement = 0.0
                vlm_prob_agreement = 0.0
                normalized_vlm_prob_agreement = 0.0
                normalized_human_prob_agreement = 0.0
                
                human_oracle_prob_agreement = 0.0
                normalized_human_oracle_prob_agreement = 0.0
                
                vlm_answer = max(vlm_probabilities.items(), key=lambda x: x[1])[0]
                assert vlm_answer in human_probabilities
                
                for ans in human_labels:
                    # remove the human's contribution from the probabilities
                    probabilities = human_probabilities.copy()
                    assert probabilities[ans] >= human_prob_contribution
                    probabilities[ans] -= human_prob_contribution
                    
                    # compute a probability distribution over the remaining answers
                    holdout_counts = {a: int(p * n_humans) for a, p in probabilities.items()}
                    Z = sum(holdout_counts.values())
                    # if P's sum is 0, set it to 1
                    if Z == 0:
                        Z = 1.0 # effect is the same anyways
                    holdout_probabilities = {a: p / Z for a, p in holdout_counts.items()}
                    # assert sum(holdout_probabilities.values()) == 1.0, f"Sum of holdout probabilities is not 1.0: {sum(holdout_probabilities.values())}"
                    
                    # use current label and check probability of agreement
                    pa = holdout_probabilities[ans]
                    vlm_pa = holdout_probabilities[vlm_answer]
                    
                    # # ORACLE
                    # # take pa to be the most common holdout probability
                    most_common_holdout_answer = max(holdout_probabilities.items(), key=lambda x: x[1])[0]
                    oracle_pa = holdout_probabilities[most_common_holdout_answer]
                    human_oracle_prob_agreement += oracle_pa
                    
                    human_prob_agreement += pa
                    vlm_prob_agreement += vlm_pa
                    
                    max_human_probability = max(holdout_probabilities.values())

                    if max_human_probability == 0.0:
                        max_human_probability = 1.0 # quick fix for now..
                    normalized_vlm_prob_agreement += vlm_pa / max_human_probability
                    normalized_human_prob_agreement += pa / max_human_probability
                    normalized_human_oracle_prob_agreement += oracle_pa / max_human_probability
                    n_pas_included += 1

                human_prob_agreement /= n_pas_included
                vlm_prob_agreement /= n_pas_included
                normalized_vlm_prob_agreement /= n_pas_included
                normalized_human_prob_agreement /= n_pas_included
                human_oracle_prob_agreement /= n_pas_included
                normalized_human_oracle_prob_agreement /= n_pas_included
                # format probabilities to a string
                vlm_probs = ', '.join([f"{k}: {v}" for k, v in vlm_probabilities.items()])
                human_probs = ', '.join([f"{k}: {v}" for k, v in human_probabilities.items()])
                human_oracle_probs = ', '.join([f"{k}: {v}" for k, v in human_probabilities.items()])

                row = {
                    "Experiment Folder": experiment_folder,
                    "Sample Folder": sample_folder,
                    "baseline_model": baseline_model,
                    "method": method,
                    "prompt_image_type": prompt_image_type,
                    "Question": question,
                    "VLM Probabilities": vlm_probs,
                    "Human Probabilities": human_probs,
                    "Human Oracle Probabilities": human_oracle_probs,
                    "VLM Probability of Agreement": vlm_prob_agreement,
                    "Human Probability of Agreement": human_prob_agreement,
                    "Human Oracle Probability of Agreement": human_oracle_prob_agreement,
                    "Normalized VLM Probability of Agreement": normalized_vlm_prob_agreement,
                    "Normalized Human Probability of Agreement": normalized_human_prob_agreement,
                    "Normalized Human Oracle Probability of Agreement": normalized_human_oracle_prob_agreement,
                    "Top-1 Accuracy": top_1_accuracy,
                    "Top-2 Accuracy": top_2_accuracy,
                    "Human Entropy": human_entropy,
                    "KL Divergence": kl_divergence,
                    "Top-1 Random Accuracy": top_1_random_accuracy,
                    "Top-2 Random Accuracy": top_2_random_accuracy,
                }
                
                if question.startswith("q_goal_position"):
                    base_question = question
                else:
                    base_question = question.rsplit('_p', 1)[0]
                
                # include sample in the cumulative metrics
                all_vlm_prob_agreement.include(vlm_prob_agreement)
                all_human_prob_agreement.include(human_prob_agreement)
                all_vlm_weighted_prob_agreement.include(normalized_vlm_prob_agreement)
                all_human_weighted_prob_agreement.include(normalized_human_prob_agreement)
                all_human_oracle_prob_agreement.include(human_oracle_prob_agreement)
                all_human_oracle_weighted_prob_agreement.include(normalized_human_oracle_prob_agreement)
                
                # include sample in the question cumulative metrics
                if base_question not in question_cumulative_vlm_prob_agreement:
                    question_cumulative_vlm_prob_agreement[base_question] = CumulativeMetric("vlm_prob_agreement")
                    question_cumulative_human_prob_agreement[base_question] = CumulativeMetric("human_prob_agreement")
                    question_cumulative_vlm_weighted_prob_agreement[base_question] = CumulativeMetric("vlm_weighted_prob_agreement")
                    question_cumulative_human_weighted_prob_agreement[base_question] = CumulativeMetric("human_weighted_prob_agreement")
                    question_cumulative_human_oracle_prob_agreement[base_question] = CumulativeMetric("human_oracle_prob_agreement")
                    question_cumulative_human_oracle_weighted_prob_agreement[base_question] = CumulativeMetric("human_oracle_weighted_prob_agreement")
                question_cumulative_vlm_prob_agreement[base_question].include(vlm_prob_agreement)
                question_cumulative_human_prob_agreement[base_question].include(human_prob_agreement)
                question_cumulative_vlm_weighted_prob_agreement[base_question].include(normalized_vlm_prob_agreement)
                question_cumulative_human_weighted_prob_agreement[base_question].include(normalized_human_prob_agreement)
                question_cumulative_human_oracle_prob_agreement[base_question].include(human_oracle_prob_agreement)
                question_cumulative_human_oracle_weighted_prob_agreement[base_question].include(normalized_human_oracle_prob_agreement)
                
                row["Base Question"] = base_question
                # Add to "All" category
                for group_name, questions in REASONING_GROUPS.items():
                    if base_question in questions:
                        reasoning_group = group_name
                        break
                else:
                    raise ValueError(f"Base question {base_question} not found in any reasoning group...")
                row["Reasoning Group"] = reasoning_group
                
                # include sample in the reasoning group cumulative metrics
                if reasoning_group not in reasoning_group_cumulative_vlm_prob_agreement:
                    reasoning_group_cumulative_vlm_prob_agreement[reasoning_group] = CumulativeMetric("vlm_prob_agreement")
                    reasoning_group_cumulative_human_prob_agreement[reasoning_group] = CumulativeMetric("human_prob_agreement")
                    reasoning_group_cumulative_vlm_weighted_prob_agreement[reasoning_group] = CumulativeMetric("vlm_weighted_prob_agreement")
                    reasoning_group_cumulative_human_weighted_prob_agreement[reasoning_group] = CumulativeMetric("human_weighted_prob_agreement")
                    reasoning_group_cumulative_human_oracle_prob_agreement[reasoning_group] = CumulativeMetric("human_oracle_prob_agreement")
                    reasoning_group_cumulative_human_oracle_weighted_prob_agreement[reasoning_group] = CumulativeMetric("human_oracle_weighted_prob_agreement")
                reasoning_group_cumulative_vlm_prob_agreement[reasoning_group].include(vlm_prob_agreement)
                reasoning_group_cumulative_human_prob_agreement[reasoning_group].include(human_prob_agreement)
                reasoning_group_cumulative_vlm_weighted_prob_agreement[reasoning_group].include(normalized_vlm_prob_agreement)
                reasoning_group_cumulative_human_weighted_prob_agreement[reasoning_group].include(normalized_human_prob_agreement)
                reasoning_group_cumulative_human_oracle_prob_agreement[reasoning_group].include(human_oracle_prob_agreement)
                reasoning_group_cumulative_human_oracle_weighted_prob_agreement[reasoning_group].include(normalized_human_oracle_prob_agreement)

                if base_question not in base_question_reasoning_group:
                    base_question_reasoning_group[base_question] = row["Reasoning Group"]

                base_question_data[base_question].append({
                    "row": row,
                    "evaluation": evaluation
                })

                if base_question not in base_question_possible_answers:
                    base_question_possible_answers[base_question] = set()
                base_question_possible_answers[base_question].update(vlm_probabilities.keys())
                base_question_possible_answers[base_question].update(human_probabilities.keys())

                if base_question not in base_question_vlm_probs:
                    base_question_vlm_probs[base_question] = {}
                for answer, probability in vlm_probabilities.items():
                    base_question_vlm_probs[base_question].setdefault(answer, []).append(probability)

                if base_question not in base_question_human_probs:
                    base_question_human_probs[base_question] = {}
                for answer, probability in human_probabilities.items():
                    base_question_human_probs[base_question].setdefault(answer, []).append(probability)

                experiment_results.append(row)

                if human_entropy > entropy_threshold:
                    experiment_disagreement_results.append(row)

                if vlm_probabilities:
                    vlm_top1_prediction = max(vlm_probabilities.items(), key=lambda x: x[1])[0]
                else:
                    vlm_top1_prediction = None

                if human_probabilities:
                    human_top1_label = max(human_probabilities.items(), key=lambda x: x[1])[0]
                else:
                    human_top1_label = None

                if vlm_top1_prediction is not None and human_top1_label is not None:
                    confusion_matrix_vlm[base_question][(vlm_top1_prediction, human_top1_label)] += 1
                    overall_confusion_matrix_vlm[(vlm_top1_prediction, human_top1_label)] += 1

                for answer, prob in human_probabilities.items():
                    count = int(prob * n_humans)
                    if human_top1_label is not None:
                        confusion_matrix_human[base_question][(answer, human_top1_label)] += count
                        overall_confusion_matrix_human[(answer, human_top1_label)] += count

                all_answers = set(vlm_probabilities.keys()).union(human_probabilities.keys())
                for human_label in human_probabilities.keys():
                    for vlm_pred in vlm_probabilities.keys():
                        if (vlm_pred, human_label) not in confusion_matrix_vlm[base_question]:
                            confusion_matrix_vlm[base_question][(vlm_pred, human_label)] += 0
                        if (vlm_pred, human_label) not in overall_confusion_matrix_vlm:
                            overall_confusion_matrix_vlm[(vlm_pred, human_label)] += 0
                    for human_pred in human_probabilities.keys():
                        if (human_pred, human_label) not in confusion_matrix_human[base_question]:
                            confusion_matrix_human[base_question][(human_pred, human_label)] += 0
                        if (human_pred, human_label) not in overall_confusion_matrix_human:
                            overall_confusion_matrix_human[(human_pred, human_label)] += 0

        # add the cumulative metrics to the experiment results, each with a row
        def add_row(full_experiment_results, name, prob_agreement, weighted_prob_agreement,
                    human_prob_agreement, weighted_human_prob_agreement, 
                    human_oracle_prob_agreement, weighted_human_oracle_prob_agreement):
            row = {
                "baseline_model": baseline_model,
                "method": method,
                "prompt_image_type": prompt_image_type,
                "Question/Category": name,
                "VLM Probability of Agreement": prob_agreement.get_average(),
                "VLM Probability of Agreement Std Error": prob_agreement.get_std_error(),
                "VLM Weighted Probability of Agreement": weighted_prob_agreement.get_average(),
                "VLM Weighted Probability of Agreement Std Error": weighted_prob_agreement.get_std_error(),
                "Human Probability of Agreement": human_prob_agreement.get_average(),
                "Human Probability of Agreement Std Error": human_prob_agreement.get_std_error(),
                "Human Weighted Probability of Agreement": weighted_human_prob_agreement.get_average(),
                "Human Weighted Probability of Agreement Std Error": weighted_human_prob_agreement.get_std_error(),
                "Human Oracle Probability of Agreement": human_oracle_prob_agreement.get_average(),
                "Human Oracle Probability of Agreement Std Error": human_oracle_prob_agreement.get_std_error(),
                "Human Oracle Weighted Probability of Agreement": weighted_human_oracle_prob_agreement.get_average(),
                "Human Oracle Weighted Probability of Agreement Std Error": weighted_human_oracle_prob_agreement.get_std_error(),
                "Number of Samples": prob_agreement.get_n_samples(),
                "Experiment Folder": experiment_folder,
            }

            full_experiment_results.append(row)
        
        # total metrics
        add_row(full_experiment_results, "All", all_vlm_prob_agreement, all_vlm_weighted_prob_agreement,
                all_human_prob_agreement, all_human_weighted_prob_agreement,
                all_human_oracle_prob_agreement, all_human_oracle_weighted_prob_agreement)
        
        # reasoning groups
        for reasoning_group in REASONING_GROUPS:
            add_row(full_experiment_results, reasoning_group, reasoning_group_cumulative_vlm_prob_agreement[reasoning_group],
                    reasoning_group_cumulative_vlm_weighted_prob_agreement[reasoning_group],
                    reasoning_group_cumulative_human_prob_agreement[reasoning_group],
                    reasoning_group_cumulative_human_weighted_prob_agreement[reasoning_group],
                    reasoning_group_cumulative_human_oracle_prob_agreement[reasoning_group],
                    reasoning_group_cumulative_human_oracle_weighted_prob_agreement[reasoning_group])
        
        for base_question in question_cumulative_vlm_prob_agreement:
            add_row(full_experiment_results, base_question, question_cumulative_vlm_prob_agreement[base_question],
                    question_cumulative_vlm_weighted_prob_agreement[base_question],
                    question_cumulative_human_prob_agreement[base_question],
                    question_cumulative_human_weighted_prob_agreement[base_question],
                    question_cumulative_human_oracle_prob_agreement[base_question],
                    question_cumulative_human_oracle_weighted_prob_agreement[base_question])

        all_results.extend(experiment_results)
        disagreement_results.extend(experiment_disagreement_results)

        # Step 4: Aggregate results per base question
        aggregated_experiment_vlm = []
        aggregated_experiment_human = []
        for base_question, data_list in base_question_data.items():
            vlm_majority_vote_accuracies = []
            vlm_top2_accuracies = []
            human_top1_accuracies = []
            human_top2_accuracies = []
            human_entropies = []
            normalized_human_entropies = []
            kl_divergences = []
            random_top1_accuracies = []
            random_top2_accuracies = []
            entropy_weighted_vlm_accuracies = []
            entropy_weighted_human_accuracies = []

            # ADDED: Lists to aggregate Probability of Agreement metrics
            vlm_prob_agreements = []
            human_prob_agreements = []
            normalized_vlm_prob_agreements = []
            normalized_human_prob_agreements = []
            human_oracle_prob_agreements = []
            normalized_human_oracle_prob_agreements = []

            num_choices = len(base_question_possible_answers.get(base_question, []))
            if num_choices < 2:
                num_choices = 2

            for item in data_list:
                row = item["row"]
                evaluation = item["evaluation"]

                vlm_majority_vote_accuracies.append(row['Top-1 Accuracy'])
                vlm_top2_accuracies.append(row['Top-2 Accuracy'])
                vlm_prob_agreements.append(row["VLM Probability of Agreement"])
                human_prob_agreements.append(row["Human Probability of Agreement"])
                human_oracle_prob_agreements.append(row["Human Oracle Probability of Agreement"])
                normalized_vlm_prob_agreements.append(row["Normalized VLM Probability of Agreement"])
                normalized_human_prob_agreements.append(row["Normalized Human Probability of Agreement"])
                normalized_human_oracle_prob_agreements.append(row["Normalized Human Oracle Probability of Agreement"])

                human_probabilities = evaluation.get("human_probabilities", {})
                sorted_probs = sorted(human_probabilities.values(), reverse=True)
                if sorted_probs:
                    human_top1_accuracies.append(sorted_probs[0])
                    human_top2_accuracies.append(sum(sorted_probs[:2]) if len(sorted_probs) > 1 else sorted_probs[0])
                else:
                    human_top1_accuracies.append(0)
                    human_top2_accuracies.append(0)

                human_entropy = row['Human Entropy']
                human_entropies.append(human_entropy)

                max_entropy = math.log2(num_choices)
                normalized_entropy = human_entropy / max_entropy if max_entropy > 0 else 0
                normalized_human_entropies.append(normalized_entropy)

                kl_divergences.append(row['KL Divergence'])
                random_top1_accuracies.append(row['Top-1 Random Accuracy'])
                random_top2_accuracies.append(row['Top-2 Random Accuracy'])

                entropy_weighted_accuracy_vlm = (1 - normalized_entropy) * vlm_majority_vote_accuracies[-1]
                entropy_weighted_vlm_accuracies.append(entropy_weighted_accuracy_vlm)

                entropy_weighted_accuracy_human = (1 - normalized_entropy) * human_top1_accuracies[-1]
                entropy_weighted_human_accuracies.append(entropy_weighted_accuracy_human)

            avg_vlm_majority_vote_accuracy = compute_average(vlm_majority_vote_accuracies)
            avg_vlm_top2_accuracy = compute_average(vlm_top2_accuracies)
            avg_human_top1_accuracy = compute_average(human_top1_accuracies)
            avg_human_top2_accuracy = compute_average(human_top2_accuracies)
            avg_human_entropy = compute_average(human_entropies)
            avg_normalized_human_entropy = compute_average(normalized_human_entropies)
            avg_kl_divergences = compute_average(kl_divergences)
            avg_random_top1_accuracy = compute_average(random_top1_accuracies)
            avg_random_top2_accuracy = compute_average(random_top2_accuracies)
            avg_entropy_weighted_vlm_accuracy = compute_average(entropy_weighted_vlm_accuracies)
            avg_entropy_weighted_human_accuracy = compute_average(entropy_weighted_human_accuracies)

            # ADDED: Compute averages for Probability of Agreement and Normalized Probability of Agreement
            avg_vlm_prob_agreement = compute_average(vlm_prob_agreements)
            avg_human_prob_agreement = compute_average(human_prob_agreements)
            avg_normalized_vlm_prob_agreement = compute_average(normalized_vlm_prob_agreements)
            avg_normalized_human_prob_agreement = compute_average(normalized_human_prob_agreements)
            avg_human_oracle_prob_agreement = compute_average(human_oracle_prob_agreements)
            avg_normalized_human_oracle_prob_agreement = compute_average(normalized_human_oracle_prob_agreements)

            cm_vlm = confusion_matrix_vlm.get(base_question, {})
            metrics_vlm = compute_metrics(cm_vlm)

            aggregated_row_vlm = {
                "Experiment Folder": experiment_folder,
                "baseline_model": baseline_model,
                "method": method,
                "prompt_image_type": prompt_image_type,
                "Base Question": base_question,
                "Reasoning Group": base_question_reasoning_group.get(base_question, "Unknown"),
                "Average VLM Majority Vote Accuracy": avg_vlm_majority_vote_accuracy,
                "Average Top-2 Accuracy": avg_vlm_top2_accuracy,
                "Average Human Entropy": avg_human_entropy,
                "Average Normalized Human Entropy": avg_normalized_human_entropy,
                "Average Entropy Weighted VLM Majority Vote Accuracy": avg_entropy_weighted_vlm_accuracy,
                "Average KL Divergence": avg_kl_divergences,
                "Average Random Top-1 Accuracy": avg_random_top1_accuracy,
                "Average Random Top-2 Accuracy": avg_random_top2_accuracy,
                "Average VLM Probability of Agreement": avg_vlm_prob_agreement,
                "Average Normalized VLM Probability of Agreement": avg_normalized_vlm_prob_agreement
            }
            aggregated_row_vlm.update(metrics_vlm)

            answer_probs = base_question_vlm_probs.get(base_question, {})
            avg_answer_probs = {answer: compute_average(probs) for answer, probs in answer_probs.items()}
            sorted_answers = sorted(avg_answer_probs.items(), key=lambda x: x[1], reverse=True)
            top_answers = sorted_answers[:5]

            for idx, (answer, avg_prob) in enumerate(top_answers, 1):
                aggregated_row_vlm[f'answer_{idx}_label'] = answer
                aggregated_row_vlm[f'answer_{idx}_prob'] = avg_prob
            for idx in range(len(top_answers)+1, 6):
                aggregated_row_vlm[f'answer_{idx}_label'] = 'N/A'
                aggregated_row_vlm[f'answer_{idx}_prob'] = 'N/A'

            aggregated_experiment_vlm.append(aggregated_row_vlm)

            cm_human = confusion_matrix_human.get(base_question, {})
            metrics_human = compute_metrics(cm_human)

            aggregated_row_human = {
                "Experiment Folder": experiment_folder,
                "Base Question": base_question,
                "Reasoning Group": base_question_reasoning_group.get(base_question, "Unknown"),
                "Average Top-1 Accuracy": avg_human_top1_accuracy,
                "Average Top-2 Accuracy": avg_human_top2_accuracy,
                "Average Human Entropy": avg_human_entropy,
                "Average Normalized Human Entropy": avg_normalized_human_entropy,
                "Average Entropy Weighted Human Majority Voice Accuracy": avg_entropy_weighted_human_accuracy,
                "Average Random Top-1 Accuracy": avg_random_top1_accuracy,
                "Average Random Top-2 Accuracy": avg_random_top2_accuracy,
                "Average Human Probability of Agreement": avg_human_prob_agreement,
                "Average Normalized Human Probability of Agreement": avg_normalized_human_prob_agreement,
                "Average Human Oracle Probability of Agreement": avg_human_oracle_prob_agreement,
                "Average Normalized Human Oracle Probability of Agreement": avg_normalized_human_oracle_prob_agreement
            }
            aggregated_row_human.update(metrics_human)
            aggregated_experiment_human.append(aggregated_row_human)
            
        # add all full_experiment_results to all_models_full_experiment_results
        all_models_full_experiment_results.extend(full_experiment_results)
            
        # now that we have all rows of full_results, we can add the cumulative metrics
        df_full_results = pd.DataFrame(full_experiment_results).round(2)
        df_full_results.to_csv(f"{experiment_folder}_full_results.csv", index=False)

        # Compute overall metrics for "All" category
        metrics_vlm_all = compute_metrics(overall_confusion_matrix_vlm)
        aggregated_row_all_vlm = {
            "Experiment Folder": experiment_folder,
            "baseline_model": baseline_model,
            "method": method,
            "prompt_image_type": prompt_image_type,
            "Base Question": "All",
            "Reasoning Group": "All",
            "Average VLM Majority Vote Accuracy": compute_average([row['Average VLM Majority Vote Accuracy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Top-2 Accuracy": compute_average([row['Average Top-2 Accuracy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Human Entropy": compute_average([row['Average Human Entropy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Normalized Human Entropy": compute_average([row['Average Normalized Human Entropy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Entropy Weighted VLM Majority Vote Accuracy": compute_average([row['Average Entropy Weighted VLM Majority Vote Accuracy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average KL Divergence": compute_average([row['Average KL Divergence'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Random Top-1 Accuracy": compute_average([row['Average Random Top-1 Accuracy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Random Top-2 Accuracy": compute_average([row['Average Random Top-2 Accuracy'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average VLM Probability of Agreement": compute_average([row['Average VLM Probability of Agreement'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All']),
            "Average Normalized VLM Probability of Agreement": compute_average([row['Average Normalized VLM Probability of Agreement'] for row in aggregated_experiment_vlm if row['Base Question'] != 'All'])
        }
        aggregated_row_all_vlm.update(metrics_vlm_all)
        for idx in range(1, 6):
            aggregated_row_all_vlm[f'answer_{idx}_label'] = 'N/A'
            aggregated_row_all_vlm[f'answer_{idx}_prob'] = 'N/A'
        aggregated_experiment_vlm.append(aggregated_row_all_vlm)

        metrics_human_all = compute_metrics(overall_confusion_matrix_human)
        aggregated_row_all_human = {
            "Experiment Folder": experiment_folder,
            "Base Question": "All",
            "Reasoning Group": "All",
            "Average Top-1 Accuracy": compute_average([row['Average Top-1 Accuracy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Top-2 Accuracy": compute_average([row['Average Top-2 Accuracy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Human Entropy": compute_average([row['Average Human Entropy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Normalized Human Entropy": compute_average([row['Average Normalized Human Entropy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Entropy Weighted Human Majority Voice Accuracy": compute_average([row['Average Entropy Weighted Human Majority Voice Accuracy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Random Top-1 Accuracy": compute_average([row['Average Random Top-1 Accuracy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Random Top-2 Accuracy": compute_average([row['Average Random Top-2 Accuracy'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Human Probability of Agreement": compute_average([row['Average Human Probability of Agreement'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Normalized Human Probability of Agreement": compute_average([row['Average Normalized Human Probability of Agreement'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Human Oracle Probability of Agreement": compute_average([row['Average Human Oracle Probability of Agreement'] for row in aggregated_experiment_human if row['Base Question'] != 'All']),
            "Average Normalized Human Oracle Probability of Agreement": compute_average([row['Average Normalized Human Oracle Probability of Agreement'] for row in aggregated_experiment_human if row['Base Question'] != 'All'])
        }
        aggregated_row_all_human.update(metrics_human_all)
        aggregated_experiment_human.append(aggregated_row_all_human)

        aggregated_results_vlm.extend(aggregated_experiment_vlm)
        aggregated_results_human.extend(aggregated_experiment_human)

        # Step 10: Write the VLM probabilities to csv
        write_vlm_probabilities_csv(root_dir, experiment_folder, baseline_model, method, prompt_image_type,
                                    base_question_possible_answers, base_question_vlm_probs, base_question_reasoning_group)

        # Step 11: Write the confusion matrices to a .txt file
        write_confusion_matrices_txt(root_dir, experiment_folder, base_question_data, confusion_matrix_vlm, confusion_matrix_human)


    df_full_results = pd.DataFrame(all_models_full_experiment_results).round(2)
    df_full_results.to_csv(f"{config['postprocessed_results_csv']}", index=False)

    # Step 6: Write the results
    write_eval_full_csv(root_dir, all_results)

    # Step 7: Write the disagreement results
    write_eval_disagreement_csv(root_dir, disagreement_results)

    # Step 8: Write the VLM aggregated results
    write_eval_aggregated_vlm_csv(root_dir, aggregated_results_vlm)

    # Step 9: Write the Human aggregated results
    write_eval_aggregated_human_csv(root_dir, aggregated_results_human)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation data from multiple experiments."
    )
    parser.add_argument(
        '--cfg_path',
        type=str,
        default='config.yaml',
        help='Path to the config file'
    )
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    evaluation_folder = config['evaluation_folder']
    survey_folder = config['survey_folder']
    compute_averages_and_generate_csv(evaluation_folder, survey_folder, entropy_threshold=0.5)
