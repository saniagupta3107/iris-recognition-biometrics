import numpy as np

# In matcher.py, replace the old function with this one

def calculate_hamming_distance(template1, template2, mask1, mask2):
    """
    Calculates the normalized Hamming Distance between two iris templates,
    using masks to exclude noisy regions.

    Args:
        template1 (np.array): The first binary iris template.
        template2 (np.array): The second binary iris template.
        mask1 (np.array): The mask corresponding to the first template.
        mask2 (np.array): The mask corresponding to the second template.

    Returns:
        float: The normalized Hamming Distance, or 1.0 if an error occurs.
    """
    if any(arg is None for arg in [template1, template2, mask1, mask2]):
        return 1.0

    # The mask needs to be compatible with the template size.
    # Since our template is 4x larger, we combine 4 masks.
    num_orientations = 4 # From our features.py
    combined_mask = np.concatenate([mask1.flatten()] * num_orientations * 2)

    # Create a final intersection mask: only compare bits where BOTH masks are valid (1)
    intersection_mask = np.logical_and(combined_mask, combined_mask)

    # Apply the mask to the templates
    masked_template1 = template1[intersection_mask]
    masked_template2 = template2[intersection_mask]
    
    if masked_template1.size == 0:
        return 1.0 # Avoid division by zero if there's no overlapping valid region

    # Calculate Hamming distance only on the valid (unmasked) bits
    distance = np.sum(np.bitwise_xor(masked_template1, masked_template2))
    normalized_distance = distance / masked_template1.size
    
    return normalized_distance