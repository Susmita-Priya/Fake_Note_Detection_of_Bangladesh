import cv2
import numpy as np
import imutils

# Load the image and convert it to grayscale
image = cv2.imread("1000tk.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to detect edges
edged = cv2.Canny(blurred, 30, 150)

# Create a single window to display the images
combined = np.hstack([gray, blurred, edged])

# Define the desired fixed height
fixed_height = 300

# Calculate the aspect ratio of the combined image
combined_aspect_ratio = combined.shape[1] / combined.shape[0]

# Calculate the width based on the fixed height
#fixed_width = int(fixed_height * combined_aspect_ratio)
# Set the maximum width for display
max_width = 1200

# Resize the combined image to the fixed width and height
if combined.shape[1] > max_width:
    combined = cv2.resize(combined, (max_width, fixed_height))

# Display the grayscale image, Gaussian blur, and edge detection in a single window
cv2.imshow("Grayscale | Gaussian Blur | Edges", combined)
cv2.waitKey(0)
