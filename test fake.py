import cv2
import numpy as np

def preprocess_image(image_path, target_size=(500, 500)):
    # Load the image directly in BGR color space
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    # Resize while preserving aspect ratio
    height, width, _ = image.shape
    if height > width:
        new_height = target_size[0]
        new_width = int(width * (new_height / height))
    else:
        new_width = target_size[1]
        new_height = int(height * (new_width / width))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Pad to reach target size
    top_pad = (target_size[0] - new_height) // 2
    bottom_pad = target_size[0] - new_height - top_pad
    left_pad = (target_size[1] - new_width) // 2
    right_pad = target_size[1] - new_width - left_pad
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    return padded_image

def detect_edges(image):
    # Detect edges using Canny in the grayscale converted image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

def compare_color_histograms(image1, image2, bins=256):
    # Calculate histogram for each channel in the BGR color space and compare
    hist1 = [cv2.calcHist([image1], [i], None, [bins], [0, 256]) for i in range(3)]
    hist2 = [cv2.calcHist([image2], [i], None, [bins], [0, 256]) for i in range(3)]
    # Normalize the histograms
    hist1 = [cv2.normalize(h, h) for h in hist1]
    hist2 = [cv2.normalize(h, h) for h in hist2]
    # Compare histograms using correlation for each channel and average the results
    similarity = sum(cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CORREL) for i in range(3)) / 3
    return similarity

def compare_with_standard(standard, test, color_threshold=0.85, edge_threshold=0.80):
    # Compare color histograms
    color_similarity = compare_color_histograms(standard, test)
    print(f"Color similarity: {color_similarity}")

    # Compare edges
    standard_edges = detect_edges(standard)
    test_edges = detect_edges(test)
    edge_similarity = np.sum(standard_edges == test_edges) / standard_edges.size
    print(f"Edge similarity: {edge_similarity}")

    # Decision based on thresholds
    return color_similarity > color_threshold and edge_similarity > edge_threshold

def main(standard_path, test_path):
    try:
        # Preprocess and potentially resize images to a common dimension
        standard_note = preprocess_image(standard_path, target_size=(500, 500))
        test_note = preprocess_image(test_path, target_size=(500, 500))

        # Compare and decide
        if compare_with_standard(standard_note, test_note):
            print("The currency is genuine.")
        else:
            print("The currency is fake.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify paths to images
standard_path = '/Users/shikharshrestha/Downloads/jpegmini_optimized-3/IMG_6250.jpg'
test_path = '/Users/shikharshrestha/Downloads/jpegmini_optimized-3/IMG_6251.jpg'
main(standard_path, test_path)
