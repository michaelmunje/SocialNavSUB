import os
import json
import math
import cv2
import numpy as np
import pytesseract
import yaml

def raise_unknown_exception(message):
    """
    We never return 'unknown' answers; if something is not determinable, 
    raise an exception.
    """
    raise ValueError(f"Cannot determine answer. Reason: {message}")

def load_config(config_filename="eval_cfg.yaml"):
    """
    Loads the YAML configuration file.
    """
    if not os.path.exists(config_filename):
        raise_unknown_exception(f"Config file '{config_filename}' not found.")
    with open(config_filename, "r") as fp:
        config = yaml.safe_load(fp)
    return config

def load_survey_prompt(filename):
    """
    Loads the survey prompt file which contains question keys and other info.
    Returns the dict.
    """
    if not os.path.exists(filename):
        raise_unknown_exception(f"Survey prompt file '{filename}' not found.")
    with open(filename, "r") as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file '{filename}': {e}")
    return data

def extract_positions_from_image(image_path, rb_cfg):
    """
    Uses OpenCV to detect colored circles and then applies OCR to the region 
    inside each circle to extract its label. Returns a dict {label: (x_norm, y_norm)},
    where (x_norm, y_norm) are normalized [0..1] coordinates.
    
    Hyperparameters for circle detection and OCR are drawn from rb_cfg.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open {image_path}")

    # Preprocessing: Convert to grayscale and apply median blur.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_ksize = rb_cfg.get("blur_ksize", 5)
    gray = cv2.medianBlur(gray, blur_ksize)

    # HoughCircles hyperparameters
    dp = rb_cfg.get("hough_dp", 1.2)
    minDist = rb_cfg.get("hough_minDist", 20)
    param1 = rb_cfg.get("hough_param1", 100)
    param2 = rb_cfg.get("hough_param2", 30)
    minRadius = rb_cfg.get("hough_minRadius", 10)
    maxRadius = rb_cfg.get("hough_maxRadius", 60)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is None or len(circles) == 0:
        raise ValueError(f"No circles detected in {image_path}")

    circles = np.round(circles[0, :]).astype(int)
    height, width = image.shape[:2]
    positions = {}

    for (x_center, y_center, radius) in circles:
        # Define ROI (region of interest) around the detected circle.
        x0 = max(0, x_center - radius)
        y0 = max(0, y_center - radius)
        x1 = min(width, x_center + radius)
        y1 = min(height, y_center + radius)
        roi = image[y0:y1, x0:x1]

        # Tesseract OCR hyperparameters
        tesseract_psm = rb_cfg.get("tesseract_psm", 8)
        tesseract_oem = rb_cfg.get("tesseract_oem", 3)
        tesseract_whitelist = rb_cfg.get("tesseract_char_whitelist", "0123456789RG")
        custom_config = f'--psm {tesseract_psm} --oem {tesseract_oem} -c tessedit_char_whitelist={tesseract_whitelist}'

        text = pytesseract.image_to_string(roi, config=custom_config)
        label = text.strip().replace("\n", "").replace(" ", "")
        if not label:
            continue

        x_norm = x_center / float(width)
        y_norm = y_center / float(height)
        positions[label] = np.array([x_norm, y_norm])

    if not positions:
        raise ValueError(f"No valid labeled circles found in {image_path}")
    return positions

def compute_robot_moving_direction(all_timesteps, rb_cfg):
    """
    Computes the robot's moving direction based on its x-coordinate change.
    
    Uses the following configurable thresholds (from rb_cfg):
      - robot_turn_threshold (default: 0.3)
      - robot_move_threshold (default: 0.1)
    """
    if len(all_timesteps) == 0:
        raise_unknown_exception("No timesteps found; cannot compute robot direction.")

    first_robot_pos = all_timesteps[0].get('R')
    last_robot_pos = all_timesteps[-1].get('R')
    if not first_robot_pos or not last_robot_pos:
        raise_unknown_exception("Robot position not found in first or last frame.")

    dx = last_robot_pos[0] - first_robot_pos[0]
    turn_threshold = rb_cfg.get("robot_turn_threshold", 0.3)
    move_threshold = rb_cfg.get("robot_move_threshold", 0.1)

    if dx <= -turn_threshold:
        return ["turning left"]
    elif dx <= -move_threshold:
        return ["moving ahead", "turning left"]
    elif dx >= turn_threshold:
        return ["turning right"]
    elif dx >= move_threshold:
        return ["moving ahead", "turning right"]
    else:
        return ["moving ahead"]

def spatial_position_of_entity_begin(all_timesteps, entity_label, spatial_offset_threshold):
    """
    Determines the spatial position (left/right/ahead) of an entity at its first appearance.
    
    Compares the entity's x-coordinate to that of the robot using spatial_offset_threshold.
    """
    robot_label = 'R'
    for t in all_timesteps:
        if entity_label in t and robot_label in t:
            ex, _ = t[entity_label]
            rx, _ = t[robot_label]
            if ex <= rx - spatial_offset_threshold:
                return "to the left of"
            elif ex >= rx + spatial_offset_threshold:
                return "to the right of"
            else:
                return "ahead of"
    raise_unknown_exception(f"Entity '{entity_label}' not found in any timestep.")

def spatial_position_of_entity_end(all_timesteps, entity_label, spatial_offset_threshold):
    """
    Determines the spatial position (left/right/ahead) of an entity at its last appearance.
    
    Compares the entity's x-coordinate to that of the robot using spatial_offset_threshold.
    """
    robot_label = 'R'
    for t in reversed(all_timesteps):
        if entity_label in t and robot_label in t:
            ex, _ = t[entity_label]
            rx, _ = t[robot_label]
            if ex <= rx - spatial_offset_threshold:
                return "to the left of"
            elif ex >= rx + spatial_offset_threshold:
                return "to the right of"
            else:
                return "ahead of"
    raise_unknown_exception(f"Entity '{entity_label}' not found in any timestep.")

def compute_person_distance_change(all_timesteps, person_label, distance_threshold):
    """
    Computes the change in distance between a person and the robot from the first to the last frame.
    
    If the distance change is within distance_threshold, it is considered "about the same".
    """
    robot_label = 'R'
    first_frame = None
    last_frame = None

    for t in all_timesteps:
        if person_label in t and robot_label in t:
            first_frame = t
            break
    for t in reversed(all_timesteps):
        if person_label in t and robot_label in t:
            last_frame = t
            break

    if not first_frame or not last_frame:
        raise_unknown_exception(f"Person '{person_label}' not found in any frames.")

    ex_start, ey_start = first_frame[person_label]
    rx_start, ry_start = first_frame[robot_label]
    ex_end, ey_end = last_frame[person_label]
    rx_end, ry_end = last_frame[robot_label]

    dist_start = math.hypot(ex_start - rx_start, ey_start - ry_start)
    dist_end = math.hypot(ex_end - rx_end, ey_end - ry_end)

    if abs(dist_end - dist_start) <= distance_threshold:
        return "about the same distance to"
    elif dist_end > dist_start + distance_threshold:
        return "further away from"
    else:
        return "closer to"

def lines_intersect(p1, p2, p3, p4):
    """
    Checks if line segment p1->p2 intersects with line segment p3->p4 using an orientation test.
    """
    def orientation(a, b, c):
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        if (o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0):
            if (o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0):
                return True
    return False

def compute_obstructing_path(all_timesteps, person_label):
    """
    Determines whether the person's trajectory obstructs the robot->goal path.
    
    Returns "yes" if the line from the first to the last position of the person 
    intersects the robot->goal line; otherwise, returns "no".
    """
    robot_label = 'R'
    goal_label = 'G'
    first_frame = None
    last_frame = None

    for t in all_timesteps:
        if person_label in t and robot_label in t and goal_label in t:
            first_frame = t
            break

    if not first_frame:
        raise_unknown_exception(f"Cannot compute obstructing path: '{person_label}' never appears with R,G together.")

    for t in reversed(all_timesteps):
        if person_label in t:
            last_frame = t
            break

    if not last_frame:
        raise_unknown_exception(f"Cannot compute obstructing path: no last frame found for '{person_label}'.")

    r_start = first_frame[robot_label]
    g_start = first_frame[goal_label]
    p_start = first_frame[person_label]
    p_end = last_frame[person_label]

    return "yes" if lines_intersect(r_start, g_start, p_start, p_end) else "no"

def distance_point_to_line(pt, line_start, line_end):
    """
    Returns the perpendicular distance from pt to the line defined by line_start->line_end.
    """
    (x0, y0) = pt
    (x1, y1) = line_start
    (x2, y2) = line_end

    denom = math.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        raise_unknown_exception("Robot->Goal line is degenerate (start=end).")
    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    return numerator / denom

def compute_obstructing_end_position(all_timesteps, person_label, threshold):
    """
    At the last frame where the person appears with R and G, determines if 
    the person is close enough to the robot->goal line (within the threshold).
    """
    robot_label = 'R'
    goal_label = 'G'
    last_frame = None

    for t in reversed(all_timesteps):
        if person_label in t and robot_label in t and goal_label in t:
            last_frame = t
            break

    if not last_frame:
        raise_unknown_exception(f"Cannot compute obstructing end position: {person_label} or R,G not found in last frame.")

    pt_p = last_frame[person_label]
    pt_r = last_frame[robot_label]
    pt_g = last_frame[goal_label]

    dist = distance_point_to_line(pt_p, pt_r, pt_g)
    return "yes" if dist <= threshold else "no"

def build_answers_for_person(all_timesteps, person_label, survey_data, rb_cfg):
    """
    For each question related to a specific person, computes the answer using
    hyperparameters from rb_cfg.
    """
    answers = {}

    spatial_threshold = rb_cfg.get("spatial_offset_threshold", 0.4)
    distance_threshold = rb_cfg.get("distance_change_threshold", 0.3)
    obstruction_threshold = rb_cfg.get("obstruction_distance_threshold", 0.1)

    # q_person_spatial_position_begin_p{i}
    begin_key = f"q_person_spatial_position_begin_p{person_label}"
    if begin_key in survey_data:
        answers[begin_key] = spatial_position_of_entity_begin(all_timesteps, person_label, spatial_threshold)

    # q_person_spatial_position_end_p{i}
    end_key = f"q_person_spatial_position_end_p{person_label}"
    if end_key in survey_data:
        answers[end_key] = spatial_position_of_entity_end(all_timesteps, person_label, spatial_threshold)

    # q_person_distance_change_p{i}
    dist_key = f"q_person_distance_change_p{person_label}"
    if dist_key in survey_data:
        answers[dist_key] = compute_person_distance_change(all_timesteps, person_label, distance_threshold)

    # q_obstructing_path_p{i} and related questions
    obstruct_key = f"q_obstructing_path_p{person_label}"
    if obstruct_key in survey_data:
        obstruction_answer = compute_obstructing_path(all_timesteps, person_label)
        answers[obstruct_key] = obstruction_answer

        aff_key = f"q_robot_affected_p{person_label}"
        if aff_key in survey_data:
            answers[aff_key] = obstruction_answer

        action_key = f"q_robot_action_p{person_label}"
        if action_key in survey_data:
            answers[action_key] = "not considering" if obstruction_answer == "no" else "avoiding"

        person_aff_key = f"q_person_affected_p{person_label}"
        if person_aff_key in survey_data:
            answers[person_aff_key] = obstruction_answer

        person_action_key = f"q_person_action_p{person_label}"
        if person_action_key in survey_data:
            answers[person_action_key] = "not considering" if obstruction_answer == "no" else "avoiding"

    # q_obstructing_end_position_p{i} and related questions
    obs_end_key = f"q_obstructing_end_position_p{person_label}"
    if obs_end_key in survey_data:
        obs_end_answer = compute_obstructing_end_position(all_timesteps, person_label, obstruction_threshold)
        answers[obs_end_key] = obs_end_answer

        sug_aff_key = f"q_robot_suggested_affected_p{person_label}"
        if sug_aff_key in survey_data:
            answers[sug_aff_key] = obs_end_answer

        sug_action_key = f"q_robot_suggested_action_p{person_label}"
        if sug_action_key in survey_data:
            answers[sug_action_key] = "not considering" if obs_end_answer == "no" else "avoiding"

        future_action_key = "q_human_future_action_prediction"
        if future_action_key in survey_data:
            answers[future_action_key] = "not considering" if obs_end_answer == "no" else "avoiding"

    return answers

def process_subfolder(subfolder_path, survey_data, rb_cfg):
    """
    Processes a single subfolder (e.g., "33_Spot_45_228") by:
      - Collecting image frames.
      - Extracting positions from each image.
      - Building answers using the rule‐based heuristic.
    """
    all_timesteps = []
    num_frames = rb_cfg.get("num_frames", 10)
    for i in range(num_frames):
        image_path = os.path.join(subfolder_path, f"sample_with_bev_{i}.png")
        if not os.path.exists(image_path):
            break
        positions = extract_positions_from_image(image_path, rb_cfg)
        all_timesteps.append(positions)

    if len(all_timesteps) == 0:
        raise_unknown_exception(f"No valid images found in {subfolder_path}.")

    answers = {}

    # Global questions:
    if "q_robot_moving_direction" in survey_data:
        answers["q_robot_moving_direction"] = compute_robot_moving_direction(all_timesteps, rb_cfg)

    spatial_threshold = rb_cfg.get("spatial_offset_threshold", 0.4)
    if "q_goal_position_begin" in survey_data:
        answers["q_goal_position_begin"] = spatial_position_of_entity_begin(all_timesteps, 'G', spatial_threshold)
    if "q_goal_position_end" in survey_data:
        answers["q_goal_position_end"] = spatial_position_of_entity_end(all_timesteps, 'G', spatial_threshold)

    # Identify person labels (exclude 'R' and 'G')
    all_labels = set()
    for t in all_timesteps:
        all_labels.update(t.keys())
    all_labels.discard('R')
    all_labels.discard('G')

    for label in all_labels:
        person_answers = build_answers_for_person(all_timesteps, label, survey_data, rb_cfg)
        answers.update(person_answers)

    return answers

def main():
    """
    Main script entry:
      1. Load eval_cfg.yaml.
      2. Extract the rule-based hyperparameters from the "rulebased_baseline" section.
      3. Load survey_prompt.json (path configurable via the top-level config).
      4. Process each subfolder in the prompts directory.
      5. Write results to a JSON file in each subfolder.
    """
    config = load_config("eval_cfg.yaml")
    # Extract hyperparameters specific to the rule-based baseline.
    rb_cfg = config.get("rulebased_baseline", {})

    prompts_dir = config.get("prompts_dir", "prompts")
    survey_prompt_path = config.get("survey_prompt", os.path.join(prompts_dir, "survey_prompt.json"))
    survey_data = load_survey_prompt(survey_prompt_path)

    for name in os.listdir(prompts_dir):
        subfolder_path = os.path.join(prompts_dir, name)
        if os.path.isdir(subfolder_path):
            try:
                results = process_subfolder(subfolder_path, survey_data, rb_cfg)
                output_filename = os.path.join(subfolder_path, f"{name}_results.json")
                with open(output_filename, "w") as out_fp:
                    json.dump(results, out_fp, indent=2)
                print(f"Processed '{subfolder_path}', results saved to '{output_filename}'")
            except Exception as e:
                raise e

if __name__ == "__main__":
    main()