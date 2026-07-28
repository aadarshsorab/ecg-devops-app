import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_ecg_roi(image):
    """Crops the ECG graph from an image by finding the largest contour."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    return image[y:y+h, x:x+w]

def remove_ecg_grid(image):
    """Removes grid lines from an ECG image using morphological operations."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    inverted = cv2.bitwise_not(gray)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    detected_horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detected_vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    detected_grid = cv2.bitwise_or(detected_horizontal, detected_vertical)
    no_grid_inverted = cv2.subtract(inverted, detected_grid)
    cleaned_image = cv2.bitwise_not(no_grid_inverted)

    return cleaned_image

def extract_signal_from_image(grid_removed_image, smoothing_window=3):
    """Extracts a 1D normalized signal from a pre-processed ECG image."""
    if len(grid_removed_image.shape) == 3:
        gray = cv2.cvtColor(grid_removed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = grid_removed_image

    inverted = cv2.bitwise_not(gray)
    height, width = inverted.shape
    signal_y_positions = np.zeros(width, dtype=np.int32)

    for x in range(width):
        column = inverted[:, x]
        y_pos = np.argmax(column)
        signal_y_positions[x] = y_pos

    raw_signal = height - signal_y_positions

    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        extracted_signal = np.convolve(raw_signal, kernel, mode='same')
    else:
        extracted_signal = raw_signal

    # Normalize 0–1
    extracted_signal = (extracted_signal - np.min(extracted_signal)) / (np.max(extracted_signal) - np.min(extracted_signal))

    return extracted_signal

def preprocess_ecg_image(image_path):
    """
    Full ECG preprocessing pipeline for a single image.
    Returns: processed_image (grid removed), extracted_signal (1D numpy array)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at: {image_path}")

    cropped = crop_ecg_roi(img)
    cleaned = remove_ecg_grid(cropped)
    signal = extract_signal_from_image(cleaned)

    return cleaned, signal
