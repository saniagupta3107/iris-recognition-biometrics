import cv2
import os
import numpy as np

def load_preprocess(path):
    """Loads an image from a path, checks if it's valid, and converts to grayscale."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # If the image cannot be read, imread returns None
    if img is None:
        return None
    
    # Convert the loaded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray
# In preprocessing.py, add this new function

def create_mask(normalized_iris):
    """
    Creates a binary mask to identify noisy regions (eyelids, eyelashes, reflections)
    in the normalized iris image. 1 = valid iris, 0 = noise.

    Args:
        normalized_iris (np.array): The unwrapped iris rectangle.

    Returns:
        np.array: A 2D binary mask of the same size as the normalized iris.
    """
    if normalized_iris is None:
        return None
    
    # Create a mask that is initially all valid (all ones)
    mask = np.ones(normalized_iris.shape, dtype=np.uint8)
    
    # Simple noise detection using intensity thresholds
    # Find very dark pixels (potential eyelashes)
    mask[normalized_iris < 50] = 0
    # Find very bright pixels (potential reflections)
    mask[normalized_iris > 230] = 0
    
    return mask

def detect_circles(gray):
    """
    Detects the pupil and iris circles in a grayscale image using the Hough Transform.
    Returns: (pupil_circle, iris_circle) tuples or (None, None) if not found.
    """
    # Using parameters tuned for the UPOL dataset
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=100,
                               param1=50,
                               param2=70,
                               minRadius=20,
                               maxRadius=120)

    # If circles are found, process them
    if circles is not None:
        # Convert circle parameters (x, y, radius) to integers
        circles = np.uint16(np.around(circles[0, :]))
        
        # Sort circles by their radius (smallest to largest)
        # The smallest is likely the pupil, the largest is the iris
        circles = sorted(circles, key=lambda c: c[2]) 
        
        if len(circles) >= 2:
            pupil = circles[0]
            iris = circles[-1]
            return tuple(pupil), tuple(iris)

    # If not enough circles are found, return None
    return None, None

# This block runs only when the script is executed directly
if __name__ == "__main__":
    # Define the relative path to the data directory
    data_dir = "../data/"
    
    # Create a list of all files in the data directory that end with .png
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    print(f"Found {len(image_files)} images to process.")

    # Loop through each image file in the list
    for filename in image_files:
        # Create the full path to the image
        path = os.path.join(data_dir, filename)
        
        print(f"Processing: {filename}")
        
        # Load and preprocess the image
        gray = load_preprocess(path)
        
        # If the image failed to load, skip to the next one
        if gray is None:
            print(f"  - Could not load {filename}, skipping.")
            continue

        # Detect the pupil and iris circles
        pupil, iris = detect_circles(gray)

        # If both pupil and iris were successfully detected
        if pupil and iris:
            print(f"  - Success! Found pupil and iris.")
            
            # Create a color image to draw on for visualization
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Draw the pupil circle (green)
            cv2.circle(output, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 2)
            
            # Draw the iris circle (red)
            cv2.circle(output, (iris[0], iris[1]), iris[2], (0, 0, 255), 2)
            
            # Display the result in a window
            cv2.imshow("Detected Iris", output)
            
            # Wait for 50 milliseconds. If 'q' is pressed, break the loop.
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            print(f"  - Failed to detect circles in {filename}")
            
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("Processing complete.")