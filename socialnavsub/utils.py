# Standard library
import base64
import csv
import io
import itertools
import json
import math
import os
import pickle
import random
import shutil
import time
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# Packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from PIL import Image

# Local
from structures import BEVPose, TrackedObject, Trajectory, localize_position_wrt_initial_pose

# Define which questions are in which reasoning group
REASONING_GROUPS = {
    "Spatial reasoning": [
        "q_person_spatial_position_begin",
        "q_person_spatial_position_end",
        "q_goal_position_begin",
        "q_goal_position_end",
        "q_obstructing_end_position"
    ],
    "Spatiotemporal reasoning": [
        "q_robot_moving_direction",
        "q_person_distance_change",
        "q_obstructing_path"
    ],
    "Social reasoning": [
        "q_robot_affected",
        "q_robot_action",
        "q_person_affected",
        "q_person_action",
        "q_robot_suggested_affected",
        "q_robot_suggested_action",
        "q_human_future_action_prediction"
    ]
}

def find_evaluation_files(root_dir):
    """
    Recursively find all "evaluation.json" files grouped by experiment folder.
    """
    experiment_files = defaultdict(list)
    for experiment_folder in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, experiment_folder)
        if os.path.isdir(experiment_path):
            for sample_folder in os.listdir(experiment_path):
                sample_path = os.path.join(experiment_path, sample_folder)
                evaluation_file = os.path.join(sample_path, 'evaluation.json')
                if os.path.isfile(evaluation_file):
                    experiment_files[experiment_folder].append(evaluation_file)
    return experiment_files


def process_evaluation_file(file_path):
    """
    Reads and returns the JSON data from the given evaluation file path.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def compute_average(values):
    """
    Compute average of a list of values.
    """
    if len(values) == 0:
        return 0
    return sum(values) / len(values)


def compute_entropy(probabilities):
    """
    Compute the entropy of a set of probabilities.
    """
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
    return entropy


def compute_kl_divergence(p, q, epsilon=1e-10):
    """
    Computes the Kullback-Leibler divergence between two probability distributions p and q.
    p and q should be dictionaries with the same set of keys.
    Missing keys or zeros in q are replaced by epsilon.
    """
    kl_divergence = 0.0
    for key in p:
        p_val = p.get(key, 0)
        q_val = q.get(key, epsilon)
        q_val = max(q_val, epsilon)  # Ensure q_val is not zero
        if p_val > 0:
            kl_divergence += p_val * math.log2(p_val / q_val)
    return kl_divergence


def compute_cohens_kappa(cm):
    """
    Compute Cohen's Kappa from a confusion matrix dictionary,
    where keys are tuples (pred, true_label) and values are counts.
    """
    total = sum(cm.values())
    if total == 0:
        return 0
    # Po: observed agreement
    sum_po = sum(cm[(label, label)] for label in set(pred for pred, true in cm.keys()) if (label, label) in cm)
    po = sum_po / total

    # Pe: expected agreement
    pred_totals = defaultdict(int)
    true_totals = defaultdict(int)
    for (pred, true_label), count in cm.items():
        pred_totals[pred] += count
        true_totals[true_label] += count
    pe = sum((pred_totals[label] * true_totals[label]) for label in pred_totals) / (total * total)
    if pe == 1:
        return 0  # Avoid division by zero
    kappa = (po - pe) / (1 - pe)
    return kappa


def compute_metrics(confusion_matrix):
    """
    Compute various metrics (Macro Precision, Recall, F1, Specificity, Cohen's Kappa)
    from a given confusion_matrix dict: {(pred, true): count}.
    """
    classes = set()
    for (pred, true_label) in confusion_matrix.keys():
        classes.add(pred)
        classes.add(true_label)
    classes = sorted(classes)
    
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []

    total_samples = sum(confusion_matrix.values())

    for c in classes:
        TP = confusion_matrix.get((c, c), 0)
        FN = sum(confusion_matrix.get((p, c), 0) for p in classes if p != c)
        FP = sum(confusion_matrix.get((c, l), 0) for l in classes if l != c)
        TN = total_samples - TP - FN - FP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        specificity_list.append(specificity)

    precision_macro = compute_average(precision_list)
    recall_macro = compute_average(recall_list)
    f1_macro = compute_average(f1_list)
    specificity_macro = compute_average(specificity_list)

    cm_count = defaultdict(int)
    for k, v in confusion_matrix.items():
        cm_count[k] = v
    kappa = compute_cohens_kappa(cm_count)

    metrics = {
        'Macro Precision': precision_macro,
        'Macro Recall': recall_macro,
        'Macro F1 Score': f1_macro,
        'Macro Specificity': specificity_macro,
        "Cohen's Kappa": kappa
    }
    return metrics


def write_eval_full_csv(root_dir, all_results):
    fieldnames = [
        'Experiment Folder', 'Sample Folder', 'baseline_model', 'method',
        'prompt_image_type', 'Question', 'Base Question', 'Top-1 Accuracy',
        'Top-2 Accuracy', 'Human Entropy', 'KL Divergence', 
        'Top-1 Random Accuracy', 'Top-2 Random Accuracy', 'Reasoning Group',
        'VLM Probabilities', 'Human Probabilities', 'Human Oracle Probabilities',
        'VLM Probability of Agreement',          # Existing
        'Human Probability of Agreement',        # Existing
        'Normalized VLM Probability of Agreement',    # NEW
        'Normalized Human Probability of Agreement',   # NEW
        'Human Oracle Probability of Agreement',
        'Normalized Human Oracle Probability of Agreement',
    ]

    with open(os.path.join(root_dir, 'eval_full.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)


def write_eval_disagreement_csv(root_dir, disagreement_results):
    fieldnames = [
        'Experiment Folder', 'Sample Folder', 'baseline_model', 'method',
        'prompt_image_type', 'Question', 'Base Question', 'Top-1 Accuracy',
        'Top-2 Accuracy', 'Human Entropy', 'KL Divergence', 
        'Top-1 Random Accuracy', 'Top-2 Random Accuracy', 'Reasoning Group',
        'VLM Probabilities', 'Human Probabilities', 'Human Oracle Probabilities',
        'VLM Probability of Agreement',          # Existing
        'Human Probability of Agreement',        # Existing
        'Normalized VLM Probability of Agreement',    # NEW
        'Normalized Human Probability of Agreement',
        'Human Oracle Probability of Agreement',
        'Normalized Human Oracle Probability of Agreement',
    ]

    with open(os.path.join(root_dir, 'eval_disagreement.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in disagreement_results:
            writer.writerow(row)


def write_eval_aggregated_vlm_csv(root_dir, aggregated_results_vlm):
    fieldnames = [
        'Experiment Folder', 'baseline_model', 'method', 'prompt_image_type',
        'Base Question', 'Reasoning Group',
        'Average VLM Majority Vote Accuracy', 'Average Top-2 Accuracy',
        'Average Random Top-1 Accuracy', 'Average Random Top-2 Accuracy', 'Average KL Divergence',
        'Average Human Entropy', 'Average Normalized Human Entropy', 'Average Entropy Weighted VLM Majority Vote Accuracy',
        "Cohen's Kappa", 'Macro Precision', 'Macro Recall', 'Macro F1 Score', 'Macro Specificity',
        'Average VLM Probability of Agreement',              # Existing
        'Average Normalized VLM Probability of Agreement'    # NEW
    ]
    for idx in range(1, 6):
        fieldnames.append(f'answer_{idx}_label')
        fieldnames.append(f'answer_{idx}_prob')

    with open(os.path.join(root_dir, 'eval_aggregated_across_question_groups_vlm.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results_vlm:
            writer.writerow(row)


def write_eval_aggregated_human_csv(root_dir, aggregated_results_human):
    fieldnames = [
        'Experiment Folder', 'Base Question', 'Reasoning Group',
        'Average Top-1 Accuracy', 'Average Top-2 Accuracy', 'Average Human Entropy',
        'Average Normalized Human Entropy', 'Average Entropy Weighted Human Majority Voice Accuracy',
        'Average Random Top-1 Accuracy', 'Average Random Top-2 Accuracy',
        "Cohen's Kappa", 'Macro Precision', 'Macro Recall', 'Macro F1 Score', 'Macro Specificity',
        'Average Human Probability of Agreement',                 # Existing
        'Average Normalized Human Probability of Agreement',
        'Average Human Oracle Probability of Agreement',
        'Average Normalized Human Oracle Probability of Agreement',
    ]

    with open(os.path.join(root_dir, 'eval_aggregated_across_question_groups_human.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results_human:
            writer.writerow(row)


def write_vlm_probabilities_csv(root_dir, experiment_folder, baseline_model, method, prompt_image_type, base_question_possible_answers,
                                base_question_vlm_probs, base_question_reasoning_group):
    # Prepare fieldnames
    fieldnames = ['Experiment Folder', 'baseline_model', 'method', 'prompt_image_type',
                  'Base Question', 'Reasoning Group']

    all_answers = set()
    for answers in base_question_possible_answers.values():
        all_answers.update(answers)
    all_answers = sorted(all_answers)
    # Map answer strings to answer labels ("answer_1", etc.)
    answer_label_mapping = {answer: f"answer_{idx}" for idx, answer in enumerate(all_answers, 1)}
    fieldnames.extend([answer_label_mapping[answer] for answer in all_answers])

    with open(os.path.join(root_dir, f"{experiment_folder}_vlm_probabilities.csv"), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for base_question, answer_probs in base_question_vlm_probs.items():
            row = {
                "Experiment Folder": experiment_folder,
                "baseline_model": baseline_model,
                "method": method,
                "prompt_image_type": prompt_image_type,
                "Base Question": base_question,
                "Reasoning Group": base_question_reasoning_group.get(base_question, "Unknown"),
            }
            for answer in all_answers:
                probs = answer_probs.get(answer, [])
                avg_prob = compute_average(probs)
                answer_label = answer_label_mapping[answer]
                row[answer_label] = avg_prob
            writer.writerow(row)


def write_confusion_matrices_txt(root_dir, experiment_folder, base_question_data, confusion_matrix_vlm, confusion_matrix_human):
    confusion_matrix_file_txt = os.path.join(root_dir, f"{experiment_folder}_confusion_matrices.txt")
    with open(confusion_matrix_file_txt, 'w') as f:
        for base_question in base_question_data.keys():
            f.write(f"Base Question: {base_question}\n\n")
            for cm_type, cm_data in [('VLM', confusion_matrix_vlm), ('Human', confusion_matrix_human)]:
                f.write(f"{cm_type} Predictions vs. Top-1 Human Labels:\n")
                cm = cm_data.get(base_question, {})
                predictions = set()
                human_labels = set()
                for (pred, true_label) in cm.keys():
                    predictions.add(pred)
                    human_labels.add(true_label)
                predictions = sorted(predictions)
                human_labels = sorted(human_labels)
                counts_matrix = [[0 for _ in human_labels] for _ in predictions]
                for i, pred in enumerate(predictions):
                    for j, human_label in enumerate(human_labels):
                        counts_matrix[i][j] = cm.get((pred, human_label), 0)
                col_widths = [max(len(label), 5) for label in human_labels]
                pred_width = max(max(len(pred) for pred in predictions), len(f"{cm_type} Predictions"))
                f.write(" " * (pred_width + 4))
                f.write("Top-1 Human Labels\n")
                f.write(f"{cm_type + ' Predictions':<{pred_width+2}}")
                for idx, human_label in enumerate(human_labels):
                    f.write(f"{human_label:<{col_widths[idx]+2}}")
                f.write("\n")
                separator_length = pred_width + 2 + sum(col_widths) + 2 * len(col_widths)
                f.write("-" * separator_length + "\n")
                for i, pred in enumerate(predictions):
                    f.write(f"{pred:<{pred_width+2}}")
                    for j in range(len(human_labels)):
                        count = counts_matrix[i][j]
                        f.write(f"{count:<{col_widths[j]+2}d}")
                    f.write("\n")
                f.write("\n")

def copy_config_files(eval_dir: str,
                      prompts_data: str,
                      samples_json_fp: str,
                      relevant_prev_qs_fp: str,
                      dataset_cfg: str):
    """
    Copies configuration and prompt-related files into the evaluation directory.
    """
    shutil.copy('eval_cfg.yaml', os.path.join(eval_dir, 'eval_cfg.yaml'))
    shutil.copy(prompts_data, os.path.join(eval_dir, 'prompts.json'))
    shutil.copy(samples_json_fp, os.path.join(eval_dir, 'sample_info.json'))
    shutil.copy(relevant_prev_qs_fp, os.path.join(eval_dir, 'relevant_prev_questions.json'))
    shutil.copy(dataset_cfg, os.path.join(eval_dir, 'dataset_cfg.yaml'))


def validate_prompts_in_human_answers(answer_dir: str, prompts: List[Tuple[str, str, List[str], str]]):
    """
    Checks if all the question keys present in prompts are found in the human answers (common_answers.json).
    """
    human_answers_fp = os.path.join(answer_dir, 'common_answers.json')
    assert os.path.exists(human_answers_fp), 'Common answers not found'
    with open(human_answers_fp, 'r') as f:
        common_answers = json.load(f)
        for question_key, _, _, _ in prompts:
            assert question_key in common_answers, f"Question key {question_key} not in human answers"


def save_debug_images(baseline_model: str, model, images_prompt: List, sample_id: str):
    """
    Saves images for debugging based on the model type and baseline_model.
    Creates a folder called 'DEBUG_LATEST_IMG_PROMPT' and saves the images.

    :param baseline_model: Name of the baseline model.
    :param model: The instantiated model with attributes like baseline_type.
    :param images_prompt: The list of images or image data (RGB numpy arrays, base64-encoded strings, or file paths).
    :param sample_id: The ID of the sample being processed.
    """
    debug_folder = "DEBUG_LATEST_IMG_PROMPT"
    os.makedirs(debug_folder, exist_ok=True)

    if baseline_model == 'dummy':
        # images_prompt are expected to be RGB or BGR numpy arrays
        for idx, img_data in enumerate(images_prompt):
            img_filename = os.path.join(debug_folder, f"{sample_id}_{idx}.jpg")
            img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_filename, img_data_rgb)
    else:
        # For API-based models, images_prompt might be base64-encoded images
        if model.baseline_type == 'api':
            for idx, img_base64 in enumerate(images_prompt):
                img_data = base64.b64decode(img_base64)
                img_filename = os.path.join(debug_folder, f"{sample_id}_{idx}.jpg")
                with open(img_filename, "wb") as img_file:
                    img_file.write(img_data)
        elif model.baseline_type == 'local' and ('llava' in baseline_model.lower() or 'spatialvlm' in baseline_model.lower()):
            # images_prompt is a list of image file paths
            for img_path in images_prompt:
                img_filename = os.path.basename(img_path)
                dst_path = os.path.join(debug_folder, f"{sample_id}_{img_filename}")
                shutil.copy(img_path, dst_path)
        else:
            raise NotImplementedError("Debug image saving not implemented for this configuration.")

def annotate_labels_for_dynamic_objects(
    img: np.ndarray,
    config: dict,
    objs_in_scene: List[TrackedObject],
    timestep: int,
    people_in_scene: dict,
    goal_pos: np.ndarray,
    time_annotations: bool = False,
    circle_rad=30,
    thickness=5,
    font_scale=1.3,
    left_offset=14):
    # in lieu of a typechecker :)
    assert isinstance(img, np.ndarray), f"Expected np.ndarray, got {type(img)} instead"
    assert isinstance(config, dict), f"Expected dict, got {type(config)} instead"
    assert isinstance(objs_in_scene, list), f"Expected list, got {type(objs_in_scene)} instead"
    
    img_size = config['native_resolution_y'], config['native_resolution_x']
    assert img.shape[:2] == img_size, "Image size does not match the native resolution of the dataset"
    h, w = img_size

    # PLOT CIRCLES FOR EACH LABEL
    for obj in objs_in_scene:
      corresponding_timestep_idx = obj.corresponding_timesteps.index(timestep)
      bb = obj.corresponding_bboxes[corresponding_timestep_idx]
      color = obj.color
      obj_idx = obj.label
      # get center pixel
      point = [int(bb[2] + bb[0]) // 2, int(bb[3] + bb[1]) // 2]
      assert 0 <= point[0] <= w and 0 <= point[1] <= h, f"Point {point} is not in the image"
      cv2.circle(img, point, circle_rad, color=color, thickness=-1)
      
      # PLOT LINES FOR PAST LOCATIONS IF time_annotations IS TRUE
      if time_annotations:
          for past_idx in range(corresponding_timestep_idx - 1, -1, -1):
              past_bb = obj.corresponding_bboxes[past_idx]
              past_point = [int(past_bb[2] + past_bb[0]) // 2, int(past_bb[3] + past_bb[1]) // 2]
              cv2.line(img, tuple(past_point), tuple(point), color=color, thickness=2)
              point = past_point
                
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)

    # PLOT NUMBERS FOR EACH LABEL
    for obj in objs_in_scene:
      corresponding_timestep_idx = obj.corresponding_timesteps.index(timestep)
      bb = obj.corresponding_bboxes[corresponding_timestep_idx]
      color = obj.color
      obj_idx = obj.label
      point = [int(bb[2] + bb[0]) // 2 - circle_rad // 2, int(bb[3] + bb[1]) // 2 + circle_rad // 2]
      
      # if 2 digits, move point to the left slightly
      if len(str(obj_idx)) > 1:
        point[0] -= left_offset
      img = cv2.putText(img, str(obj_idx), point, font,
                    font_scale, font_color, thickness, cv2.LINE_AA)
      # also add to people_in_scene
      track_id = obj.id
      people_in_scene[track_id] = 1
      
    k1 = config["dist_coeffs"]["k1"] if "dist_coeffs" in config else 0.0
    k2 = config["dist_coeffs"]["k2"] if "dist_coeffs" in config else 0.0
    p1 = config["dist_coeffs"]["p1"] if "dist_coeffs" in config else 0.0
    p2 = config["dist_coeffs"]["p2"] if "dist_coeffs" in config else 0.0
    k3 = config["dist_coeffs"]["k3"] if "dist_coeffs" in config else 0.0
    dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])
    
    camera_matrix = get_camera_matrix(config)

    if goal_pos is None:
        return img

    # make copy of the img
    img = img.copy()
    # let's also annotate the goal position onto the image
    goal_pos = get_pos_pixels(
        np.array([goal_pos]),
        config['camera_height'],
        config['camera_x_offset'],
        camera_matrix,
        dist_coeffs,
        img_size,
        clip=True,
    )[0]
    black_color = (0, 0, 0)
    cv2.circle(img, (int(goal_pos[0]), int(goal_pos[1])), circle_rad, black_color, -1)
    point = [int(goal_pos[0]) - circle_rad // 2, int(goal_pos[1]) + circle_rad // 2]
    point = [min(max(point[0], 0), h), min(max(point[1], 0), w)]
    point = [int(goal_pos[0]) - circle_rad // 2, int(goal_pos[1]) + circle_rad // 2]
    
    cv2.putText(img, "G", point, font, font_scale, font_color, thickness, cv2.LINE_AA)    

    return img

def get_endpoint_diff(start, yaw, length):
    # Calculate the direction vector (dx, dy)
    dx = np.cos(yaw)
    dy = np.sin(yaw)
    
    # Normalize the direction vector to ensure its Euclidean length is 1
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm
    
    # resize so that the arrow has a fixed length
    dx *= length
    dy *= length
    
    return dx, dy

def plot_object_bev_past(ax, 
                        object_num, 
                        obj, 
                        object_x_pos, 
                        object_y_pos, 
                        object_yaws, 
                        object_pos_diff, 
                        color, 
                        fixed_arrow_length=0.25,
                        linewidth=2):
    """
    Plots an object on the scene.

    Parameters:
    - ax: Matplotlib axis object.
    - object_num: Identifier number of the object.
    - x, y: Coordinates of the object.
    - timestep: Time step for the arrow direction.
    - color: Color of the object and arrow.
    """
    timestep = obj.corresponding_timesteps[-1]
    corresponding_timestep_idx = obj.corresponding_timesteps.index(timestep)
    x = object_x_pos[corresponding_timestep_idx]
    y = object_y_pos[corresponding_timestep_idx]
    yaw = object_yaws[corresponding_timestep_idx]
    mag_diff_next_timestep = object_pos_diff[corresponding_timestep_idx]
    
    for past_idx in range(corresponding_timestep_idx - 1, -1, -1):
        past_x = object_x_pos[past_idx]
        past_y = object_y_pos[past_idx]
        # plot line from past to current
        plt.plot([past_y, y], [past_x, x], color=color, linewidth=linewidth)
        x, y = past_x, past_y

def plot_object_bev(ax, 
                        object_num, 
                        obj, 
                        object_x_pos, 
                        object_y_pos, 
                        object_yaws, 
                        object_pos_diff, 
                        timestep, 
                        color, 
                        fixed_arrow_length=0.25,
                        markersize=15,
                        fontsize=10,
                        arrow_width=0.05):
    """
    Plots an object on the scene.

    Parameters:
    - ax: Matplotlib axis object.
    - object_num: Identifier number of the object.
    - x, y: Coordinates of the object.
    - timestep: Time step for the arrow direction.
    - color: Color of the object and arrow.
    """
    
    corresponding_timestep_idx = obj.corresponding_timesteps.index(timestep)
    x = object_x_pos[corresponding_timestep_idx]
    y = object_y_pos[corresponding_timestep_idx]
    yaw = object_yaws[corresponding_timestep_idx]
    mag_diff_next_timestep = object_pos_diff[corresponding_timestep_idx]
    
    ax.plot(y, x, 'o', color=color, markersize=markersize)
    ax.text(y, x, str(object_num), ha='center', va='center', color='white', fontsize=fontsize)
    if mag_diff_next_timestep > 0.075:
        end_point = get_endpoint_diff([x, y], yaw, fixed_arrow_length)
        ax.arrow(y, x, end_point[1], end_point[0], width=arrow_width, fc=color, ec=color)

def plot_trajectory_bev(ax, 
                            x, 
                            y, 
                            yaw, 
                            previous_wps, 
                            color='black', 
                            time_annotations: bool = False,
                            fixed_arrow_length=0.25,
                            arrow_width=0.05,
                            traj_markersize=15,
                            traj_fontsize=10,
                            line_width=2):
    """
    Draws an arrow over the timesteps to show trajectory.

    Parameters:
    - ax: Matplotlib axis object.
    - x, y: Coordinates of the trajectory.
    - color: Color of the trajectory arrow.
    """
    ax.plot(y, x, 'o', color=color, markersize=traj_markersize)
    ax.text(y, x, 'R', ha='center', va='center', color='white', fontsize=traj_fontsize)
    end_point = get_endpoint_diff([x, y], yaw, fixed_arrow_length)
    ax.arrow(y, x, end_point[1], end_point[0], width=arrow_width, fc=color, ec=color)
    
    if time_annotations:
        # plot lines between previous waypoints
        # iterate backwards
        current_point_x, current_point_y = x, y
        for i in range(len(previous_wps) - 1, -1, -1):
            prev_x, prev_y = previous_wps[i]
            ax.plot([prev_y, current_point_y], [prev_x, current_point_x], color=color, linewidth=line_width)
            current_point_x, current_point_y = prev_x, prev_y

def generate_bev_view(objs_in_scene, 
                           wps_and_yaws, 
                           previous_wps,
                           timestep, 
                           goal_pos, 
                           axis_limits,
                           time_annotations: bool = False,
                           markersize=15,
                           fontsize=10, 
                           arrow_width=0.05,
                           fixed_arrow_length=0.25,
                           linewidth=2):
    x_min, x_max, y_min, y_max = axis_limits
    fig, ax = plt.subplots(figsize=(7, 7))
    
    if time_annotations:
        for obj in objs_in_scene:
            tracking_id = obj.id
            obj_label = obj.label
            b, g, r = obj.color
            color = (r / 255.0, g / 255.0, b / 255.0)
            color_hex = matplotlib.colors.to_hex(color)
            object_x_pos = [pose.x for pose in obj.trajectory.bev_poses]
            object_y_pos = [pose.y for pose in obj.trajectory.bev_poses]
            object_yaws = [pose.yaw for pose in obj.trajectory.bev_poses]
            object_pos_diff = [diff for diff in obj.position_differences]
            # for this timestep
            
            # filter out s.t. only past timesteps are included
            past_timesteps = [t for t in obj.corresponding_timesteps if t <= timestep]
            if len(past_timesteps) == 0:
                continue
            object_x_pos = []
            object_y_pos = []
            object_yaws = []
            object_pos_diff = []
            for past_timestep in range(len(past_timesteps)):
                past_x = obj.trajectory.bev_poses[past_timestep].x
                past_y = obj.trajectory.bev_poses[past_timestep].y
                past_yaw = obj.trajectory.bev_poses[past_timestep].yaw
                past_diff = obj.position_differences[past_timestep]
                
                object_x_pos.append(past_x)
                object_y_pos.append(past_y)
                object_yaws.append(past_yaw)
                object_pos_diff.append(past_diff)
            past_trajectory = Trajectory(bev_poses=[BEVPose(x, y, yaw) for x, y, yaw in zip(object_x_pos, object_y_pos, object_yaws)],
                                        corresponding_timesteps=past_timesteps,
                                        possible_timesteps=obj.possible_timesteps,
                                        id=obj.id,
                                        localize=False,
                                        initial_yaw_estimation=False)
            corresponding_bboxes = [obj.corresponding_bboxes[obj.corresponding_timesteps.index(t)] for t in past_timesteps]
            # make new object with only past timesteps 
            obj = TrackedObject(trajectory=past_trajectory,
                                corresponding_timesteps=past_timesteps,
                                possible_timesteps=obj.possible_timesteps,
                                id=obj.id,
                                position_differences=object_pos_diff,
                                corresponding_bboxes=corresponding_bboxes)
            
            plot_object_bev_past(ax,
                                obj_label,
                                obj,
                                object_x_pos,
                                object_y_pos,
                                object_yaws,
                                object_pos_diff,
                                color_hex,
                                fixed_arrow_length=fixed_arrow_length,
                                linewidth=linewidth)

    for obj in objs_in_scene:
        tracking_id = obj.id
        obj_label = obj.label
        b, g, r = obj.color
        color = (r / 255.0, g / 255.0, b / 255.0)
        color_hex = matplotlib.colors.to_hex(color)
        object_x_pos = [pose.x for pose in obj.trajectory.bev_poses]
        object_y_pos = [pose.y for pose in obj.trajectory.bev_poses]
        object_yaws = [pose.yaw for pose in obj.trajectory.bev_poses]
        object_pos_diff = [diff for diff in obj.position_differences]
        
        if timestep == -1:
            timestep_to_use = obj.corresponding_timesteps[-1]
        else:
            timestep_to_use = timestep
            
        # get corresponding idx
        plot_object_bev(ax, 
                            obj_label, 
                            obj, 
                            object_x_pos, 
                            object_y_pos, 
                            object_yaws, 
                            object_pos_diff, 
                            timestep_to_use, 
                            color_hex,
                            fixed_arrow_length=fixed_arrow_length,
                            markersize=markersize,
                            fontsize=fontsize,
                            arrow_width=arrow_width)
        # make sure within axis bounds
        # assert x_min <= x <= x_max, f"Object x not within bounds: {y}, bounds are {x_min} and {x_max}"
        # assert y_min <= y <= y_max, f"Object y not within bounds: {x}, bounds are {y_min} and {y_max}"

    # only include xys and ys from current timestep
    x = wps_and_yaws[0]
    y = wps_and_yaws[1]
    yaw = wps_and_yaws[2]
    plot_trajectory_bev(ax, 
                            x, 
                            y, 
                            yaw,
                            previous_wps=previous_wps,
                            time_annotations=time_annotations,
                            fixed_arrow_length=fixed_arrow_length,
                            arrow_width=arrow_width,
                            traj_markersize=markersize,
                            traj_fontsize=fontsize,
                            line_width=linewidth)
    # make sure within axis bounds
    assert x_min <= x <= x_max, f"Robot x not within bounds: {x}, bounds are {x_min} and {x_max}"
    assert y_min <= y <= y_max, f"Robot y not within bounds: {y}, bounds are {y_min} and {y_max}"
    
    if goal_pos is not None:
        # make sure within axis bounds
        assert x_min <= goal_pos[0] <= x_max, f"Goal x not within bounds: {goal_pos[0]}, bounds are {x_min} and {x_max}"
        assert y_min <= goal_pos[1] <= y_max, f"Goal y not within bounds: {goal_pos[1]}, bounds are {y_min} and {y_max}"
        ax.plot(goal_pos[1], goal_pos[0], 'o', color='black', markersize=markersize)
        ax.text(goal_pos[1], goal_pos[0], 'G', ha='center', va='center', color='white', fontsize=fontsize)
    
    
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    # create fp based on timestamp
    timestamp = int(time.time())
    fp = f'tmp_{timestamp}.png'
    plt.savefig(fp, dpi=300)
    plt.close(fig)
    img = cv2.imread(fp)
    os.remove(fp)
    return img

def get_axis_limits(objs: Dict[str, TrackedObject], robot_waypoints, goal_pos=None):
    
    # goal should be completely visible
    x_min = 0
    x_max = 10
    y_min = -5
    y_max = 5
    
    buffer_zone = 0.5
    for track_id in objs:
        x_min = min(x_min, float(objs[track_id].closest_x) - buffer_zone)
        x_max = max(x_max, float(objs[track_id].furthest_x) + buffer_zone)
        y_min = min(y_min, float(objs[track_id].closest_y) - buffer_zone)
        y_max = max(y_max, float(objs[track_id].furthest_y) + buffer_zone)
        
    xs = [wp[0] for wp in robot_waypoints]
    ys = [wp[1] for wp in robot_waypoints]
    
    # make sure every point of the trajectory is visible
    x_min = min(x_min, min(xs) - buffer_zone)
    x_max = max(x_max, max(xs) + buffer_zone)
    y_min = min(y_min, min(ys) - buffer_zone)
    y_max = max(y_max, max(ys) + buffer_zone)
        
    if goal_pos is not None:
        x_min = min(x_min, goal_pos[0] - buffer_zone)
        x_max = max(x_max, goal_pos[0] + buffer_zone)
        y_min = min(y_min, goal_pos[1] - buffer_zone)
        y_max = max(y_max, goal_pos[1] + buffer_zone)

    return (x_min, x_max, y_min, y_max)

def get_annotated_and_bev(imagepath, config, goal_pos, n_waypoints_ahead,
                               circle_rad=30,
                                thickness=5,
                                font_scale=1.3,
                                left_offset=14,
                                markersize=15,
                                fontsize=10, 
                                arrow_width=0.05,
                                fixed_arrow_length=0.25,
                                linewidth=2,
                                include_time_annotations_img=False,
                                include_time_annotations_bev=False):
    # get the image folder
    localized_objs_dir = config['lart_localized_scene_objects']
    # also get future imgs
    imgs_in_sequence = []
    future_img_fps = []
    base_img_idx = int(imagepath.split('/')[-1].split('.')[0])
    assert base_img_idx >= 0, f"Base image index is less than 0: {base_img_idx}"
    
    for i in range(n_waypoints_ahead):
        future_img_fp = imagepath.replace(f'{base_img_idx}.jpg', f'{base_img_idx + i}.jpg')
        if not os.path.exists(future_img_fp):
            print(f"Future image does not exist: {future_img_fp}, skipping...")
            return None, None, None, None, None
        future_img_fps.append(future_img_fp)
        imgs_in_sequence.append(cv2.imread(future_img_fp))
        
    localized_info_fp = get_autolabel_base_filepath(imagepath, localized_objs_dir, config) + '_localized_obj_coords.json'
    with open(localized_info_fp, 'r') as f:
        localized_info = json.load(f)
            
    track_id_to_object_trajs = localized_info['track_id_to_object_trajectory']
    possible_timesteps = [timestep for timestep in range(len(imgs_in_sequence))]

    # object_labels = [obj for obj in obj_infos if obj['timestep'] == 0]
    # for each tracking id, create a trajectory
    trackable_objects: Dict[str, TrackedObject] = {}
    for track_id in track_id_to_object_trajs:
        object_history = track_id_to_object_trajs[track_id]
        # get all objects with this tracking id
        bev_poses = [BEVPose(float(obj['center_x']), float(obj['center_y']), float(obj['yaw'])) for obj in object_history]
        corresponding_timesteps = [obj['timestep'] for obj in object_history]
        corresponding_bboxes = [obj['bbox'] for obj in object_history]
        position_differences = [obj['pos_diff'] for obj in object_history]
        if len(corresponding_timesteps) == 1:
            bev_poses[0].yaw = 0.0
        obj_trajectory = Trajectory(bev_poses=bev_poses, 
                                    corresponding_timesteps=corresponding_timesteps,
                                    possible_timesteps=possible_timesteps,
                                    id=track_id,
                                    localize=False,
                                    initial_yaw_estimation=True)

        trackable_obj = TrackedObject(trajectory=obj_trajectory, 
                                      corresponding_timesteps=corresponding_timesteps,
                                      possible_timesteps=possible_timesteps,
                                      id=track_id,
                                      position_differences=position_differences,
                                      corresponding_bboxes=corresponding_bboxes)
        trackable_objects[track_id] = trackable_obj
        
    filtered_trackable_objects: Dict[str, TrackedObject] = {}
    for track_id in trackable_objects:
        if -5 <= trackable_objects[track_id].closest_y <= 5 and 0 <= trackable_objects[track_id].closest_x <= 10 \
                and (trackable_objects[track_id].n_timesteps_appeared >= n_waypoints_ahead // 3):
            filtered_trackable_objects[track_id] = trackable_objects[track_id]
        if -3 <= trackable_objects[track_id].closest_y <= 3 and 0 <= trackable_objects[track_id].closest_x <= 5:
            filtered_trackable_objects[track_id] = trackable_objects[track_id]
    if len(filtered_trackable_objects) == 0:
        print(f"No objects in scene: {imagepath}, skipping...")
        return None, None, None, None, None

    # get list of track_ids sorted by bbox position
    track_ids_by_appearance = sorted(filtered_trackable_objects, key=lambda x: filtered_trackable_objects[x].corresponding_bboxes[0][0])

    # get number of unique labels
    n_labels = len(track_ids_by_appearance)
    
    tracking_id_to_obj_label = {track_id: i + 1 for i, track_id in enumerate(track_ids_by_appearance)}
    for track_id in filtered_trackable_objects:
        filtered_trackable_objects[track_id].assign_label(tracking_id_to_obj_label[track_id])
            
    # save tracking ids used to file
    tracking_id_to_obj_label_fp = imagepath.replace('.jpg', '_tracking_id_to_obj_label.json')
    with open(tracking_id_to_obj_label_fp, 'w') as f:
        json.dump(tracking_id_to_obj_label, f)
    
    colormap = plt.get_cmap('brg')
    colors = colormap(np.linspace(0, 1, n_labels))
    colors = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b, _ in colors]
    tracking_id_to_color = {key: color for key, color in zip(track_ids_by_appearance, colors)}
    people_in_scene = {} # track_id -> 1
    
    for track_id in tracking_id_to_color:
        filtered_trackable_objects[track_id].assign_color(tracking_id_to_color[track_id])
        
    
    future_wps, future_yaws, initial_wp, initial_yaw = get_future_and_current_waypoints(imagepath, n_waypoints_ahead)
    if future_wps is None:
        return None, None, None, None, None
    
    assert future_wps is not None, f"Future waypoints are None: {imagepath}"
    
    future_wps, future_yaws = get_future_waypoints_localized(imagepath, n_waypoints_ahead)
    if future_wps is None:
        return None, None, None, None, None
    
    axis_limits = get_axis_limits(filtered_trackable_objects, future_wps, goal_pos)
    
    timestep_to_objs_in_scene = {i: [] for i in range(len(imgs_in_sequence))}
    all_objs_in_scene = []
    for track_id in filtered_trackable_objects:
        object_history = filtered_trackable_objects[track_id]
        all_objs_in_scene.append(object_history)
        for timestep in range(len(imgs_in_sequence)):
            if timestep in object_history.corresponding_timesteps:
                corresponding_timestep_idx = object_history.corresponding_timesteps.index(timestep)
                # make sure it's within axis limits too at the timestep
                x_at_timestep = object_history.trajectory.bev_poses[corresponding_timestep_idx].x
                y_at_timestep = object_history.trajectory.bev_poses[corresponding_timestep_idx].y
                if axis_limits[0] <= x_at_timestep <= axis_limits[1] and axis_limits[2] <= y_at_timestep <= axis_limits[3]:
                    timestep_to_objs_in_scene[timestep].append(object_history)

    # save gif of annotated_imgs
    all_imgs = []
    all_bevs = []
    prev_yaw = None
    for timestep in range(len(imgs_in_sequence)):
        img = imgs_in_sequence[timestep]
        # obj must be in this timestep
        future_objs = timestep_to_objs_in_scene[timestep]
        
        
        # only include wps that are up to this timestep
        future_wps_curr = future_wps[timestep]
        future_yaws_curr = future_yaws[timestep]
        # remove samples that are too far in the future
        # timestep = i
        # estimate yaw based on next timestep
        x, y = future_wps_curr
        if timestep < len(future_wps) - 3:
            next_x, next_y = future_wps[timestep + 3]
            next_yaw = np.arctan2(next_y - y, next_x - x)
            prev_yaw = next_yaw
        elif timestep < len(future_wps) - 2:
            next_x, next_y = future_wps[timestep + 2]
            next_yaw = np.arctan2(next_y - y, next_x - x)
            prev_yaw = next_yaw
        elif timestep < len(future_wps) - 1:
            next_x, next_y = future_wps[timestep + 1]
            next_yaw = np.arctan2(next_y - y, next_x - x)
            prev_yaw = next_yaw
        else:
            next_yaw = prev_yaw
        
        robot_x, robot_y = future_wps_curr
        robot_yaw = future_yaws_curr

        adjusted_goal_pos = localize_position_wrt_initial_pose(
            future_position=np.array(goal_pos),
            initial_position=np.array([robot_x, robot_y]),
            initial_yaw=next_yaw
        )
        
        future_annotated_img = annotate_labels_for_dynamic_objects(img=img, 
                                                                   config=config, 
                                                                   objs_in_scene=future_objs, 
                                                                   timestep=timestep,
                                                                   time_annotations=include_time_annotations_img,
                                                                   people_in_scene=people_in_scene,
                                                                   goal_pos=None,
                                                                   circle_rad=circle_rad,
                                                                   thickness=thickness,
                                                                   font_scale=font_scale,
                                                                   left_offset=left_offset)
        all_imgs.append(future_annotated_img)
        
        
        wps_and_yaws = [x, y, next_yaw]
        
        previous_wps = []
        for i in range(timestep):
            previous_wps.append(future_wps[i])
        
        bev_img_with_goal = generate_bev_view(objs_in_scene=future_objs, 
                                                        wps_and_yaws=wps_and_yaws,
                                                        previous_wps=previous_wps, 
                                                        timestep=timestep, 
                                                        time_annotations=include_time_annotations_bev,
                                                        goal_pos=goal_pos, 
                                                        axis_limits=axis_limits,
                                                        markersize=markersize,
                                                        fontsize=fontsize, 
                                                        arrow_width=arrow_width,
                                                        fixed_arrow_length=fixed_arrow_length,
                                                        linewidth=linewidth)
        all_bevs.append(bev_img_with_goal)
    # for the last one, add time annotations
    img = img.copy()
    img_with_time_annotated = annotate_labels_for_dynamic_objects(img=img, 
                                                                  config=config, 
                                                                  objs_in_scene=future_objs, 
                                                                  timestep=timestep,
                                                                  people_in_scene=people_in_scene,
                                                                  goal_pos=None,
                                                                  time_annotations=include_time_annotations_img,
                                                                   circle_rad=circle_rad,
                                                                   thickness=thickness,
                                                                   font_scale=font_scale,
                                                                   left_offset=left_offset)
    # for bev here, we want to include all objs in the scene across time
    
    bev_with_time_annotated = generate_bev_view(objs_in_scene=all_objs_in_scene,
                                                            wps_and_yaws=wps_and_yaws,
                                                            previous_wps=previous_wps,
                                                            timestep=-1,
                                                            goal_pos=goal_pos,
                                                            axis_limits=axis_limits,
                                                            time_annotations=include_time_annotations_bev,
                                                            markersize=markersize,
                                                            fontsize=fontsize, 
                                                            arrow_width=arrow_width,
                                                            fixed_arrow_length=fixed_arrow_length,
                                                            linewidth=linewidth)
    n_people = len(people_in_scene)
    return all_imgs, all_bevs, img_with_time_annotated, bev_with_time_annotated, n_people

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    
def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def load_yaml(yaml_file: str):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def load_model_class(baseline_model: str, 
                     model_to_api_key: dict = None,
                     use_cot: bool = False,
                     quantization_bits: int = None):
    """
    Load the appropriate baseline model class based on the model name.
    
    :param baseline_model: The name of the baseline model to load.
    :param api_key_env_var: Optional API key environment variable for GPT-4 based models.
    :param use_cot: Whether to use chain-of-thought reasoning.
    :param quantization_bits: Quantization precision for Hugging Face models (4-bit or 8-bit).
    :return: An instance of the model class.
    """
    if ('gpt' in baseline_model.lower()) or ('o1' in baseline_model.lower()) or ('o3' in baseline_model.lower()) or ('o4' in baseline_model.lower()):
        from gpt4o import GPT4o as model_class
        api_key_env_var = model_to_api_key.get('gpt4o')
        api_key_env_var = api_key_env_var if api_key_env_var else ''
        if api_key_env_var:
            model = model_class(model_name=baseline_model,
                                api_key_env_var=api_key_env_var)
        else:
            model = model_class()
    elif 'gemini' in baseline_model.lower():
        from gemini import Gemini as model_class
        api_key_env_var = model_to_api_key.get('gemini')
        api_key_env_var = api_key_env_var if api_key_env_var else ''
        if api_key_env_var:
            model = model_class(model_name=baseline_model,
                                api_key_env_var=api_key_env_var)
        else:
            model = model_class()
    elif baseline_model.lower() == 'spatialvlm':
        from spatialvlm import SpatialVLMBaseline
        model = SpatialVLMBaseline(model_name='remyxai/SpaceLLaVA', 
                                   use_cot=use_cot)
    elif baseline_model.lower() == 'llava':
        from llava import LLaVaBaseline
        model = LLaVaBaseline(model_name='llava-hf/llava-1.5-13b-hf', 
                              use_cot=use_cot,
                              quantization_bits=quantization_bits)
    elif baseline_model.lower() == 'llava-video':
        from llava import LLaVaBaseline
        model = LLaVaBaseline(model_name="llava-hf/LLaVA-NeXT-Video-7B-hf", 
                              use_cot=use_cot,
                              quantization_bits=quantization_bits)
    elif baseline_model.lower() == 'dummy':
        from dummy import DummyBaseline
        model = DummyBaseline()
    else:
        raise ValueError('Invalid baseline model')
    
    return model

# from (multiple vlm queries) or (multiple human labelers)
class Answer:
    def __init__(self, answers: List[str], answers_probabilities: List[float], choices: List[str], n_choices: int, question_type: str):
        self.question_type = question_type
        if self.question_type == 'multiple_select':
            # powerset of answers, but at least 1 answer
            self.n_choices = 2 ** n_choices - 1
            assert self.n_choices > 0, 'No answer choices'
            assert self.n_choices > n_choices, 'Invalid n choices for multiple select'
            self.choices: List[Tuple[str]] = list(chain.from_iterable(combinations(choices, r) for r in range(1, len(choices) + 1)))
            assert len(self.choices) == self.n_choices, 'Invalid number of choices'
        elif self.question_type == 'multiple_choice':
            self.n_choices = n_choices
            self.choices: List[str] = choices
        else:
            raise ValueError('Invalid question type')

        self.answers = answers
        self.answers_probabilities = answers_probabilities
        
        self.answer_to_probability = {choice: 0.0 for choice in self.choices}
        
        adjusted_answers = []
        
        for original_answer, prob in zip(answers, answers_probabilities):
            if self.question_type == 'multiple_select':
                if isinstance(original_answer, str):
                    answer = (original_answer,)
                else:
                    answer = tuple(original_answer)
                ## assert answer in self.choices, f'Invalid answer, {answer} not in {self.choices}'
            else:
                answer = original_answer
            adjusted_answers.append(answer)
            if answer in self.choices:
                self.answer_to_probability[answer] = prob
            else:
                print(f"Warning: {answer} not in {self.choices}, will be marked as incorrect.") # \
        self.answers = adjusted_answers
        
    def get_most_common_answer(self):
        return max(self.answer_to_probability, key=self.answer_to_probability.get)
    
    def get_random_answer(self):
        return random.choice(self.choices)

def compute_cross_entropy(vlm_answer: Answer, human_answer: Answer, epsilon=1e-10):
    total = 0.0
    for choice in vlm_answer.choices:
        assert choice in human_answer.answer_to_probability, 'Invalid choice'
        assert choice in vlm_answer.answer_to_probability, 'Invalid choice'
        vlm_prob = np.clip(vlm_answer.answer_to_probability[choice], epsilon, 1)
        human_prob = human_answer.answer_to_probability[choice]
        total += human_prob * np.log2(vlm_prob)
    return -total

def compute_top_k_accuracy(vlm_answer: Answer, human_answer: Answer, k: int):
    # Get the top-k human choices by probability
    top_k_human_choices = sorted(
        human_answer.choices,
        key=lambda choice: human_answer.answer_to_probability[choice],
        reverse=True
    )[:k]
    
    # if k == 1 and top probability is 0.5 AND the second highest probability is 0.5
    # then include both in the top k
    if k == 1 and human_answer.answer_to_probability[top_k_human_choices[0]] == 0.5:
        second_highest = sorted(
            human_answer.choices,
            key=lambda choice: human_answer.answer_to_probability[choice],
            reverse=True
        )[1]
        if human_answer.answer_to_probability[second_highest] == 0.5:
            top_k_human_choices.append(second_highest)
            
    # make sure top_k probabilities are greater than 0
    top_k_human_choices = [choice for choice in top_k_human_choices if human_answer.answer_to_probability[choice] > 0]
    assert all([human_answer.answer_to_probability[choice] > 0 for choice in top_k_human_choices]), 'Invalid top k human probabilities'
    
    total_probability = 0.0
    for choice in top_k_human_choices:
        # Accumulate the VLM's probability for these choices
        total_probability += vlm_answer.answer_to_probability.get(choice, 0.0)
    
    return total_probability

def compute_top_k_human_accuracy(human_answer: Answer, k: int):
    # if human answers are a list of lists, convert to a tuple
    # if human_answer.question_type == 'multiple_select':
    #     human_answer.answers = [tuple(answer) for answer in human_answer.answers]
    
    # Get the top-k human choices by probability
    top_k_human_choices = sorted(human_answer.choices, 
                                 key=lambda choice: human_answer.answer_to_probability[choice], 
                                 reverse=True)[:k]
    
    total_probability = 0.0
    for choice in top_k_human_choices:
        assert choice in human_answer.answer_to_probability, 'Invalid choice'
        if choice in human_answer.answers:
            total_probability += human_answer.answer_to_probability[choice]
            
    assert total_probability > 0, f"Total probability is 0 for {human_answer.answers}"
    return total_probability

def load_human_answer(human_answers_json_fp: str, question_key: str, choices: List[str], question_type: str) -> Answer:
    with open(human_answers_json_fp, 'r') as f:
        raw_human_answers = json.load(f)
        raw_human_answer = raw_human_answers[question_key]
        raw_human_answer_probabilities = raw_human_answers[f'{question_key}_probabilities']
        # make suree at least 1 answer has probability > 0
        assert any([prob > 0 for prob in raw_human_answer_probabilities]), f'No answer has probability > 0 for {question_key}'
        return Answer(raw_human_answer, raw_human_answer_probabilities, choices, len(choices), question_type)

RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
CYAN = np.array([0, 1, 1])
MAGENTA = np.array([1, 0, 1])

def convert_to_global_coords(
    local_coords: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    """
    Convert local coordinates to global coordinates.

    Args:
        local_coords: Local coordinates.
        curr_pos: Current position.
        curr_yaw: Current yaw.

    Returns:
        Global coordinates.
    """
    rotmat = yaw_rotmat(curr_yaw)
    if local_coords.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif local_coords.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return local_coords.dot(rotmat) + curr_pos


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    config: dict,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
    fig = None,
    ax = None):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.

    Args:
        obs_img: image of the observation
        goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        goal_pos: goal position in the image
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """
    
    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])
    if len(pred_waypoints.shape) > 2:
        trajs = [*pred_waypoints, label_waypoints]
        traj_colors = [CYAN for _ in range(len(pred_waypoints))] + [MAGENTA]
        traj_labels = [f"prediction {i + 1}" for i in range(len(pred_waypoints))] + ["ground truth"]
    else:
        trajs = [pred_waypoints, label_waypoints]
        traj_colors = [CYAN, MAGENTA]
        traj_labels = ["prediction", "ground truth"]
    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=traj_colors,
        traj_labels=traj_labels,
    )
    
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        config,
        trajs,
        [start_pos, goal_pos],
        traj_colors=traj_colors,
    )
    ax[2].imshow(goal_img)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    # fig.clear()
    if not display:
        plt.close(fig)

def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    img_size: tuple,
    clip: Optional[bool] = False):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    print('gelloooo')
    pixels = project_points(
        points, camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = img_size[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, img_size[0]),
                    np.clip(p[1], 0, img_size[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [img_size[0], img_size[1]])
            ]
        )
    return pixels

def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])

def get_img_with_traj_overlay(
    obs_img,
    config: dict,
    goal_pos: np.ndarray,
    label_waypoints: np.ndarray,
    object_labels: list,
    fig = None,
    ax = None):
    
    # in lieu of a typechecker :)
    assert isinstance(obs_img, np.ndarray), f"Expected np.ndarray, got {type(obs_img)} instead"
    assert isinstance(config, dict), f"Expected dict, got {type(config)} instead"
    assert isinstance(goal_pos, np.ndarray), f"Expected np.ndarray, got {type(goal_pos)} instead"
    assert isinstance(label_waypoints, np.ndarray), f"Expected np.ndarray, got {type(label_waypoints)} instead"
    assert isinstance(object_labels, list), f"Expected list, got {type(object_labels)} instead"

    fig, ax = plt.subplots(figsize=(16, 9))
    start_pos = np.array([0, 0])
    trajs = [label_waypoints]
    traj_colors = [MAGENTA]
    traj_labels = ["ground truth"]

    img, track_id_to_obj_idx = plot_trajs_and_points_on_image(
        ax,
        obs_img,
        config,
        trajs,
        [start_pos, goal_pos],
        object_labels,
        traj_colors=traj_colors,
    )

    # save_path = sample_idx + '_traj_overlay.jpg'
    # plt.axis('off')
    # fig.savefig(save_path, bbox_inches="tight")
    # plt.close(fig)
    # img = cv2.imread(save_path)
    return img, track_id_to_obj_idx

def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    img_size: tuple,
    clip: Optional[bool] = False):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points, camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = img_size[1] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, img_size[1]),
                    np.clip(p[1], 0, img_size[0]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [img_size[1], img_size[0]])
            ]
        )
    return pixels

def local_coordinate_to_global(x, y, current_pos, current_yaw, initial_wp, initial_yaw):
    """
    Converts local coordinates to global coordinates.
    
    Parameters:
    - x, y: Local coordinates.
    - current_pos: Current position of the vehicle.
    - current_yaw: Current yaw of the vehicle.
    
    Returns:
    - x, y: Global coordinates.
    """
    coords = [x, y]
    delta_yaw = (current_yaw - initial_yaw)
    initial_x, initial_y = initial_wp
    curr_x, curr_y = current_pos
    
    delta_x = curr_x - initial_x
    delta_y = curr_y - initial_y
    
    R = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)], 
                    [np.sin(delta_yaw), np.cos(delta_yaw)]])
    
    R2 = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], 
                    [-np.sin(initial_yaw), np.cos(initial_yaw)]])
    
    T = np.array([delta_x, delta_y])

    coords = np.array(coords)
    global_coords = R @ coords[:2] + (R2 @ T)
    return global_coords

def get_future_waypoints_localized(image_path: str, waypoints_ahead: int):
    """
    Get future waypoints.

    Args:
        image_path: Path of the image.
        flipped: Flag indicating if the image is flipped.

    Returns:
        Tuple containing the goal label and VLM label.
    """
    future_wps, future_yaws, current_wp, current_yaw = get_future_and_current_waypoints(image_path, waypoints_ahead)
    
    if future_wps is None:
        return None, None
    
    if len(future_wps) < waypoints_ahead:
        return None, None
    
    assert len(future_wps) >= waypoints_ahead, f"Number of waypoints ahead is less than {waypoints_ahead}"
    
    future_wps_localized = to_local_coords(future_wps, current_wp, current_yaw)
    future_yaws_localized = future_yaws - current_yaw + np.pi / 2

    assert len(future_wps_localized) >= waypoints_ahead, f"Number of waypoints ahead is less than {waypoints_ahead}"

    future_wps_localized = future_wps_localized[:waypoints_ahead]
    future_yaws_localized = future_yaws_localized[:waypoints_ahead]
    return future_wps_localized, future_yaws_localized

def get_future_and_current_waypoints(image_path: str, waypoints_ahead: int):
    """
    Get future waypoints.

    Args:
        image_path: Path of the image.
        flipped: Flag indicating if the image is flipped.

    Returns:
        Tuple containing the goal label and VLM label.
    """
    folder = os.path.dirname(image_path)
    json_filepath = os.path.join(folder, "traj_data.pkl")
    assert os.path.exists(json_filepath), f"Trajectory data does not exist: {json_filepath}"
    
    data = None
    with open(json_filepath, "rb") as f:
        data = pickle.load(f)

    assert data is not None, f"Trajectory data is None: {json_filepath}"
    assert "position" in data, f"Trajectory data does not contain 'position' key: {json_filepath}"
    assert "yaw" in data, f"Trajectory data does not contain 'yaw' key: {json_filepath}"
    
    waypoints = data["position"]
    yaws = data["yaw"]

    traj_number = int(os.path.basename(image_path).split(".")[0])
    assert traj_number >= 0, f"Trajectory number is less than 0: {traj_number}"
    assert traj_number < len(waypoints), f"Trajectory number is greater than number of waypoints: {traj_number}"
    
    is_one_of_last_images = len(waypoints) < traj_number + waypoints_ahead
    if is_one_of_last_images:
        return None, None, None, None

    current_wp = waypoints[traj_number]
    current_yaw = yaws[traj_number]

    future_wps = waypoints[traj_number + 1 :]
    future_yaws = yaws[traj_number + 1 :]
    
    return future_wps, future_yaws, current_wp, current_yaw

def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    
    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])

def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing

def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    config: dict,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN]):
    """
    Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
    """
    assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    

    img_size = config['native_resolution_y'], config['native_resolution_x']
    assert img.shape[:2] == img_size, "Image size does not match the native resolution of the dataset"

    # resize to img_size = (640, 480)
    # print(img.shape)
    # img = cv2.resize(img, img_size)
    # print(img.shape)
    # cv convert to rgb
    ax.imshow(img)
    
    camera_height = config["camera_height"] if "camera_height" in config else 0.0
    camera_x_offset = config["camera_x_offset"] if "camera_x_offset" in config else 0.0
    camera_matrix = get_camera_matrix(config)

    k1 = config["dist_coeffs"]["k1"] if "dist_coeffs" in config else 0.0
    k2 = config["dist_coeffs"]["k2"] if "dist_coeffs" in config else 0.0
    p1 = config["dist_coeffs"]["p1"] if "dist_coeffs" in config else 0.0
    p2 = config["dist_coeffs"]["p2"] if "dist_coeffs" in config else 0.0
    k3 = config["dist_coeffs"]["k3"] if "dist_coeffs" in config else 0.0
    dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

    for i, traj in enumerate(list_trajs):
        xy_coords = traj[:, :2]  # (horizon, 2)
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, img_size, clip=False
        )
        if len(traj_pixels.shape) == 2:
            ax.plot(
                traj_pixels[:250, 0],
                traj_pixels[:250, 1],
                color=traj_colors[i],
                lw=2.5,
            )

    for i, point in enumerate(list_points):
        if len(point.shape) == 1:
            # add a dimension to the front of point
            point = point[None, :2]
        else:
            point = point[:, :2]
        pt_pixels = get_pos_pixels(
            point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, img_size, clip=True
        )
        ax.plot(
            pt_pixels[:250, 0],
            pt_pixels[:250, 1],
            color=point_colors[i],
            marker="o",
            markersize=10.0,
        )
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, img_size[1] - 0.5))
        ax.set_ylim((img_size[0] - 0.5, 0.5))

def get_camera_matrix(config, half_res=False):
    """
    Get the camera matrix from the config file.

    Args:
        config (dict): The config for the dataset

    Returns:
        np.ndarray: The camera matrix
        
    """
    fx, fy = config['camera_matrix']['fx'], config['camera_matrix']['fy']
    cx, cy = config['camera_matrix']['cx'], config['camera_matrix']['cy']
    if half_res:
        fx /= 2
        fy /= 2
        cx /= 2
        cy /= 2
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

def process_compressed_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image
    """
    return Image.open(io.BytesIO(msg.data))

def get_autolabel_base_filepath(imagepath: str, data_save_dir: str, config: dict):
    """
    Get the base filepath for autolabeling based on the given imagepath, data_save_dir, and config.

    Args:
        imagepath (str): The path of the image file.
        data_save_dir (str): The directory where the autolabeling data will be saved.
        config (dict): The config for the dataset

    Returns:
        str: The base filepath for autolabeling.
        
    # Example:
    #     >>> config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    #     >>> imagepath = '/home/user/data/frames/000000.jpg'
    #     >>> data_save_dir = '/home/user/data/autolabeling'
    #     >>> get_autolabel_base_filepath(imagepath, data_save_dir, config)
    #     '/home/user/data/autolabeling/000000'
    """
    frames_dir = config['input_frames_directory']
    frames_dir = os.path.abspath(frames_dir)
    imagepath = os.path.abspath(imagepath)
    imagepath = imagepath.replace(frames_dir, '')
    imagepath = imagepath.replace('.jpg', '')
    return data_save_dir + '/' + imagepath

def load_imagepaths(config):
    folder = config['input_frames_directory']
    imagepaths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                imagepaths.append(os.path.join(root, file))
    return imagepaths    

def load_image_llava(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
