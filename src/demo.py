import os
import numpy as np
import cv2

# Import all the necessary functions from your modules
from preprocessing import load_preprocess, detect_circles, create_mask
from features import normalize_iris, extract_features
from matcher import calculate_hamming_distance

# You might need to adjust this threshold after improving the features
MATCH_THRESHOLD = 0.38 

def enroll_person(person_name, image_path, templates_dir="../templates"):
    """
    Processes an iris, creates a template and a mask, and saves them together.
    """
    print(f"--- Enrolling {person_name} ---")
    
    os.makedirs(templates_dir, exist_ok=True)

    # 1. Preprocessing
    gray_image = load_preprocess(image_path)
    if gray_image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    pupil, iris = detect_circles(gray_image)
    if pupil is None or iris is None:
        print("Error: Could not detect iris and pupil.")
        return
    print("Step 1/3: Preprocessing successful.")

    # 2. Feature and Mask Creation
    normalized = normalize_iris(gray_image, pupil, iris)
    template = extract_features(normalized)
    mask = create_mask(normalized)
    
    if template is None or mask is None:
        print("Error: Could not extract features or create mask.")
        return
    print("Step 2/3: Feature extraction and masking successful.")

    # 3. Save Template and Mask together in one file
    # We use a dictionary to store both arrays
    data_to_save = {'template': template, 'mask': mask}
    template_path = os.path.join(templates_dir, f"{person_name}.npy")
    np.save(template_path, data_to_save)
    
    print(f"Step 3/3: Template and mask saved successfully to {template_path}")
    print("-" * 20)

def verify_person(live_image_path, enrolled_template_path):
    """
    Compares a live iris against a saved template and mask.
    """
    print(f"--- Verifying image {os.path.basename(live_image_path)} ---")
    
    if not os.path.exists(enrolled_template_path):
        print(f"Error: Enrolled template not found at {enrolled_template_path}")
        print("-" * 20)
        return

    # 1. Generate template and mask for the live image
    gray_image = load_preprocess(live_image_path)
    if gray_image is None:
        print(f"Could not process live image: {live_image_path}")
        return

    pupil, iris = detect_circles(gray_image)
    if pupil is None:
        print(f"Could not find iris in live image: {live_image_path}")
        return
        
    normalized = normalize_iris(gray_image, pupil, iris)
    live_template = extract_features(normalized)
    live_mask = create_mask(normalized)

    if live_template is None or live_mask is None:
        print("Could not generate a template from the live image.")
        return

    # 2. Load the enrolled template and mask
    # allow_pickle=True is needed to load a dictionary
    enrolled_data = np.load(enrolled_template_path, allow_pickle=True).item()
    enrolled_template = enrolled_data['template']
    enrolled_mask = enrolled_data['mask']

    # 3. Match templates using their masks
    distance = calculate_hamming_distance(live_template, enrolled_template, live_mask, enrolled_mask)
    print(f"Hamming Distance (Masked): {distance:.4f}")

    # 4. Make decision
    if distance < MATCH_THRESHOLD:
        print(f"Result: MATCH! (Distance is below threshold {MATCH_THRESHOLD})")
    else:
        print(f"Result: NO MATCH! (Distance is above threshold {MATCH_THRESHOLD})")
    print("-" * 20)


if __name__ == "__main__":
    # Use the same good example images as before
    enroll_image_file = "../data/036R_1.png"
    verify_image_good = "../data/036R_2.png"
    verify_image_bad = "../data/047L_1.png"

    # --- DEMO ---
    
    # 1. Enroll "Person_036"
    enroll_person("Person_036", enroll_image_file)

    # 2. Verify against another image of the same person
    enrolled_template_file = "../templates/Person_036.npy"
    verify_person(verify_image_good, enrolled_template_file)
    
    # 3. Verify against an image of a different person
    verify_person(verify_image_bad, enrolled_template_file)