import numpy as np
import cv2
from skimage.filters import gabor

def normalize_iris(gray_image, pupil_circle, iris_circle):
    """
    Unwraps the circular iris region into a rectangular block (Daugman's Rubber Sheet model).

    Args:
        gray_image (np.array): The grayscale eye image.
        pupil_circle (tuple): A tuple (x, y, r) for the pupil.
        iris_circle (tuple): A tuple (x, y, r) for the iris.

    Returns:
        np.array: The normalized 2D iris rectangle, or None if inputs are invalid.
    """
    if pupil_circle is None or iris_circle is None:
        return None

    radial_resolution = 64
    angular_resolution = 512

    pupil_x, pupil_y, pupil_r = pupil_circle
    iris_x, iris_y, iris_r = iris_circle

    theta = np.linspace(0, 2 * np.pi, angular_resolution)
    radius = np.linspace(pupil_r, iris_r, radial_resolution)

    radius_grid, theta_grid = np.meshgrid(radius, theta)

    # Convert polar coordinates to Cartesian, ensuring the data type is float32
    # THIS IS THE FIX: Changed .astype(float) to .astype(np.float32)
    x_cart = (iris_x + radius_grid * np.cos(theta_grid)).astype(np.float32)
    y_cart = (iris_y + radius_grid * np.sin(theta_grid)).astype(np.float32)

    normalized_iris = cv2.remap(gray_image, x_cart, y_cart, cv2.INTER_LINEAR)

    return normalized_iris

# In features.py, replace the old function with this one

def extract_features(normalized_iris):
    """
    Extracts a binary iris code by applying a bank of Gabor filters
    at different orientations.

    Args:
        normalized_iris (np.array): The unwrapped iris rectangle.

    Returns:
        np.array: A 1D binary vector representing the concatenated iris code.
    """
    if normalized_iris is None:
        return None

    total_iris_code = []
    # Define Gabor filter parameters
    frequency = 0.1
    # Create a bank of filters at 4 different orientations
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0°, 45°, 90°, 135°

    for theta in orientations:
        # Apply the Gabor filter for the current orientation
        real_part, imag_part = gabor(normalized_iris, frequency=frequency, theta=theta)

        # Encode the filter response into a 2-bit binary code
        # 1st bit is from the real part, 2nd bit is from the imaginary part
        code = np.zeros(real_part.shape, dtype=np.uint8)
        code[real_part > 0] = 1
        code[imag_part > 0] += 2 # Add 2 to make 4 distinct values (00, 01, 10, 11)
        
        # We can flatten this code or use it as is
        # For this example, let's create a simple binary vector
        iris_code_segment = np.zeros((real_part.shape[0], real_part.shape[1] * 2), dtype=np.uint8)
        iris_code_segment[:, ::2] = (real_part > 0).astype(np.uint8)
        iris_code_segment[:, 1::2] = (imag_part > 0).astype(np.uint8)
        
        total_iris_code.append(iris_code_segment.flatten())

    # Concatenate the codes from all filter orientations into one long feature vector
    return np.concatenate(total_iris_code)