import json
from typing import List, Dict, Callable, Tuple

import os
import json
import pickle
import tqdm
import random
import sys
import argparse
import shutil
import re
import ast
import copy
import base64
from typing import Optional, List, Tuple, Dict
import yaml

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import yaml

from structures import BEVPose, Trajectory, TrackedObject, localize_position_wrt_initial_pose
from utils import get_autolabel_base_filepath
from utils import get_camera_matrix, convert_to_global_coords, project_points, get_pos_pixels
from utils import RED, GREEN, CYAN, MAGENTA, get_future_and_current_waypoints, get_future_waypoints_localized
from utils import get_annotated_and_bev

def get_image_prompt(config_filepath: str, sample_filename: str, prompt_config: Dict):

    config = yaml.load(open(config_filepath, "r"), Loader=yaml.FullLoader)
    n_waypoints_ahead = 10
    image_fp = sample_filename
    # 81_Spot_0/1082.jpg -> 81_Spot_0_1082/1082.jpg
    image_id = image_fp.split('/')[-1].split('.')[0]
    image_id = image_fp.split('/')[-2] + '_' + image_id
    image_fp = os.path.join(image_id, image_fp.split('/')[-1])
    
    # make sure image_fp is in the corresponding directory
    parent_dir = prompt_config['frontview_folder']
    image_fp = os.path.join(parent_dir, image_fp)
    
    sample_id = image_id

    # You might need to provide the goal position or modify this part as per your data structure
    # For now, we'll assume a default goal position
    # go into prompts folder and get goal info.json
    sample_context_fp = os.path.join(prompt_config['data_folder'], 'goal_info.json')
    get_sample_context = json.load(open(sample_context_fp, "r"))
    goal_imagepath, raw_goal_pos = get_sample_context[sample_filename]
    g_x_g_y = raw_goal_pos.replace(' ', '').replace('[', '').replace(']', '').split(',')
    goal_pos = np.array([int(g_x_g_y[0]) / 1000.0, int(g_x_g_y[1]) / 1000.0])
    # convert goal

    all_imgs, all_bevs, img_with_time_annotated, bev_with_time_annotated, n_people = get_annotated_and_bev(
        image_fp, sample_id, prompt_config, goal_pos, n_waypoints_ahead,
        circle_rad=prompt_config['circle_rad_annotated_img'],
        thickness=prompt_config['thickness_annotated_img'],
        font_scale=prompt_config['font_scale_annotated_img'],
        left_offset=prompt_config['left_offset_annotated_img'],
        markersize=prompt_config['markersize_bev'],
        fontsize=prompt_config['fontsize_bev'], 
        arrow_width=prompt_config['arrow_width_bev'],
        fixed_arrow_length=prompt_config['fixed_arrow_length_bev'],
        linewidth=prompt_config['linewidth_bev'],
        include_time_annotations_img=prompt_config['include_time_annotations_img'],
        include_time_annotations_bev=prompt_config['include_time_annotations_bev'],
    )

    if all_imgs is None:
        return None

    # Extract four images: first, two from the middle, and last
    # indices = [0, 3, 6, 9]
    # extracted_imgs = [all_imgs[i] for i in indices]
    
    extracted_imgs = [img for img in all_imgs]
    indices = list(range(len(all_imgs)))
    
    # # **Add time annotations to images**
    # for idx, img in enumerate(extracted_imgs):
    #     # Define the text
    #     text = f't={idx+1}'
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 3
    #     thickness = 3
    #     text_color = (255, 255, 255)  # White text
    #     border_color = (0, 0, 0)  # Black border
        
    #     # Get the text size
    #     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    #     text_width, text_height = text_size
    #     baseline = text_size[1] + 10
        
    #     # Define the position of the text
    #     x = img.shape[1] - text_width - 20  # Adjust as needed
    #     y = 100  # Adjust as needed for the vertical position

    #     # Draw a filled rectangle (black border) behind the text
    #     top_left = (x - 10, y - text_height - 10)
    #     bottom_right = (x + text_width + 10, y + baseline - 10)
    #     cv2.rectangle(img, top_left, bottom_right, border_color, -1)

    #     # Put the text on top of the rectangle
    #     cv2.putText(
    #         img,
    #         text,
    #         (x, y),
    #         font,
    #         font_scale,
    #         text_color,
    #         thickness,
    #         cv2.LINE_AA
    #     )

    # Convert images to PIL format
    pil_imgs = []
    for img in extracted_imgs:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_imgs.append(pil_img)

    # Get dimensions of images
    width, height = pil_imgs[0].size

    # Create a new image for the grid
    grid_img = Image.new('RGB', (width * 2, height * 2))

    # Paste images into the grid
    grid_img.paste(pil_imgs[0], (0, 0))                # Top-left
    grid_img.paste(pil_imgs[1], (width, 0))            # Top-right
    grid_img.paste(pil_imgs[2], (0, height))           # Bottom-left
    grid_img.paste(pil_imgs[3], (width, height))       # Bottom-right

    # Convert top-down annotated image to PIL format
    bev_with_time_annotated_pil = Image.fromarray(cv2.cvtColor(bev_with_time_annotated, cv2.COLOR_BGR2RGB))

    # Resize the top-down image to have the same width as the grid image
    grid_width, grid_height = grid_img.size
    bev_width, bev_height = bev_with_time_annotated_pil.size
    new_bev_height = int((grid_width / bev_width) * bev_height)
    resized_bev_img = bev_with_time_annotated_pil.resize((grid_width, new_bev_height))

    # Create a new image to hold both the grid and the top-down image
    total_height = grid_height + new_bev_height
    grid_with_bev_image = Image.new("RGB", (grid_width, total_height), (255, 255, 255))  # White background

    # Paste the grid image at the top
    grid_with_bev_image.paste(grid_img, (0, 0))

    # Paste the resized top-down image at the bottom
    grid_with_bev_image.paste(resized_bev_img, (0, grid_height))

    # Create combined images from all images and top-down views
    combined_images = []
    for annotated_img, bev_img in zip(all_imgs, all_bevs):
        main_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        bev_img_pil = Image.fromarray(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))

        # Resize the top-down image to have the same width as the main image
        main_width, main_height = main_img.size
        bev_width, bev_height = bev_img_pil.size
        new_bev_height = int((main_width / bev_width) * bev_height)
        resized_bev_img = bev_img_pil.resize((main_width, new_bev_height))

        # Create a new image to hold both images
        total_height = main_height + new_bev_height
        combined_img = Image.new("RGB", (main_width, total_height), (255, 255, 255))

        # Paste images
        combined_img.paste(main_img, (0, 0))
        combined_img.paste(resized_bev_img, (0, main_height))

        combined_images.append(combined_img)
        
    # Make sure all images are in the correct format
    pil_imgs = [np.array(img) for img in pil_imgs]
    grid_img = np.array(grid_img)
    bev_with_time_annotated = np.array(bev_with_time_annotated)
    grid_with_bev_image = np.array(grid_with_bev_image)
    all_imgs = [np.array(img) for img in all_imgs]
    all_bevs = [np.array(img) for img in all_bevs]
    combined_images = [np.array(img) for img in combined_images]
    
    # Extract images from the indices
    all_imgs = [all_imgs[i] for i in indices]
    all_bevs = [all_bevs[i] for i in indices]
    combined_images = [combined_images[i] for i in indices]

    if prompt_config['prompt_img_type'] == 'img_with_bev':
        return combined_images
    elif prompt_config['prompt_img_type'] == 'grid_with_bev':
        return [grid_with_bev_image]
    elif prompt_config['prompt_img_type'] == 'grid':
        return [grid_img]
    elif prompt_config['prompt_img_type'] == 'img':
        return pil_imgs
    elif prompt_config['prompt_img_type'] == 'bev':
        return all_bevs
    else:
        raise ValueError(f"Invalid prompt_img_type: {prompt_config['prompt_img_type']}")

def get_prompt(question_key, person_idx, instructions_key, survey_info: Dict, relevant_prev_qs: dict = None, include_cot: bool = False) -> Tuple[str, str, List[str], str]:
    survey_instructions = '\n'.join(survey_info["survey_instructions"])
    no_ppl_qs = ['q_robot_moving_direction', 'q_goal_position_begin', 'q_goal_position_end']
    prompt = survey_instructions + '\n'
    
    if include_cot:
        # check to see if prev questions are relevant
        assert relevant_prev_qs is not None, 'Relevant previous questions not provided'
        assert question_key in relevant_prev_qs, f'Question key {question_key} not found in relevant previous questions'
        relevant_prev_qs = relevant_prev_qs[question_key]
        for prev_q_key in relevant_prev_qs:
            assert prev_q_key in survey_info, f'Previous question {prev_q_key} not found in survey info'
            prev_q = survey_info[prev_q_key]['question']
            prev_q = survey_info[prev_q_key]["question"].replace('{PERSON}', f'person {person_idx}')
            prev_questions = survey_info[prev_q_key]['choices']
            if person_idx > 0 and not prev_q_key in no_ppl_qs:
                prev_q_key_updated = prev_q_key + f'_p{person_idx}'
            else:
                prev_q_key_updated = prev_q_key
            prompt += f'{prev_q}\n'
            # prompt += f'Possible answers: {", ".join(prev_questions)}\n'
            # add dummy answer that we will replace later
            dummy_answer = "{\"answer\": {PREV_Q_ANSWER}"
            dummy_answer = dummy_answer.replace("PREV_Q", prev_q_key_updated)
            prompt += dummy_answer + '\n'
            
    prompt += '\n' + '\n'.join(survey_info[instructions_key]) + '\n'
    
    choices_txt = ', '.join([f'"{choice}"' for choice in survey_info[question_key]["choices"]])
    q = survey_info[question_key]["question"].replace('{PERSON}', f'person {person_idx}')
    if q.startswith('person'):
        q = 'P' + q[1:]
    answers_txt = 'Possible answers: ' + choices_txt

    answers_format_mc = survey_info['answers_format_mc']
    answers_format_ms = survey_info['answers_format_ms']
    
    # if question is multiple select, let them know they can select multiple answers, otherwise single answer
    question_type = survey_info[question_key]["type"]
    if question_type == "multiple_choice":
        answers_txt += answers_format_mc
    else:
        answers_txt += answers_format_ms
        
    choices = survey_info[question_key]["choices"]
    
    if person_idx > 0:
        question_key = question_key + f'_p{person_idx}'
        
    prompt += '\n' + q + '\n' + answers_txt

    return (question_key, prompt, choices, question_type)

def load_survey_questions(survey_info: Dict, n_pedestrians: int, relevant_prev_qs: dict = None, include_cot: bool = False) -> List[str]:
    # questions need to include the survey instructions and specific instructions for that question
    questions: List[str] = []
    
    # movement questions
    questions.append(get_prompt('q_robot_moving_direction', 0, "movement_instructions", survey_info, 
                                relevant_prev_qs=relevant_prev_qs, 
                                include_cot=include_cot))
    for i in range(1, n_pedestrians + 1):
        questions.append(get_prompt(f'q_person_spatial_position_begin', i, "movement_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs, 
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_person_spatial_position_end', i, "movement_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_person_distance_change', i, "movement_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        
    # goal location questions
    questions.append(get_prompt('q_goal_position_begin', 0, "goal_location_instructions", survey_info, 
                                relevant_prev_qs=relevant_prev_qs,
                                include_cot=include_cot))
    questions.append(get_prompt('q_goal_position_end', 0, "goal_location_instructions", survey_info, 
                                relevant_prev_qs=relevant_prev_qs,
                                include_cot=include_cot))
    for i in range(1, n_pedestrians + 1):
        questions.append(get_prompt(f'q_obstructing_path', i, "goal_location_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_obstructing_end_position', i, "goal_location_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        
    # navigation action questions
    for i in range(1, n_pedestrians + 1):
        questions.append(get_prompt(f'q_robot_affected', i, "navigation_affected_instructions", survey_info, 
                                relevant_prev_qs=relevant_prev_qs,
                                include_cot=include_cot))
        questions.append(get_prompt(f'q_robot_action', i, "navigation_action_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_person_affected', i, "navigation_affected_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_person_action', i, "navigation_action_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        
    # suggested future navigation action questions
    for i in range(1, n_pedestrians + 1):
        questions.append(get_prompt(f'q_robot_suggested_affected', i, "suggested_future_navigation_affected_instructions", survey_info, 
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_robot_suggested_action', i, "suggested_future_navigation_action_instructions", survey_info,
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        questions.append(get_prompt(f'q_human_future_action_prediction', i, "suggested_future_navigation_action_instructions", survey_info,
                                    relevant_prev_qs=relevant_prev_qs,
                                    include_cot=include_cot))
        
    return questions
    
def load_survey_questions_independent(survey_filepath: str, n_pedestrians: int) -> List[str]:
    with open(survey_filepath, 'r') as file:
        survey_info = json.load(file)
    return load_survey_questions(survey_info, n_pedestrians)

def load_survey_questions_cot(survey_filepath: str, n_pedestrians: int, relevant_prev_qs: dict) -> List[str]:
    with open(survey_filepath, 'r') as file:
        survey_info = json.load(file)
    return load_survey_questions(survey_info, n_pedestrians, relevant_prev_qs, include_cot=True)

def load_survey_questions_cot_with_gt(survey_filepath: str, n_pedestrians: int, relevant_prev_qs: dict) -> List[str]:
    with open(survey_filepath, 'r') as file:
        survey_info = json.load(file)
    return load_survey_questions(survey_info, n_pedestrians, relevant_prev_qs, include_cot=True)