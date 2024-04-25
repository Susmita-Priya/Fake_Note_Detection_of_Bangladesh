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

# Find contours in the edged image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Initialize the flag for fake note detection
fake_note_detected = False

# Loop over the contours
for c in cnts:
    # Approximate the contour to a polygon
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If the contour has four points, it may be a rectangle
    if len(approx) == 4:
        # Calculate the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # If the aspect ratio is approximately 1, it may be a Bangladeshi note
        if ar >= 0.95 and ar <= 1.05:
            # Calculate the area of the contour
            contour_area = cv2.contourArea(c)

            # Define the threshold value for fake notes
            fake_threshold = 5000  # Adjust this threshold as needed

            if contour_area > fake_threshold:
                # The note is classified as real
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "Real Note", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # The note is classified as fake
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "Fake Note", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            fake_note_detected = True
            break

# If no fake note is detected, draw a red rectangle around the entire image
if not fake_note_detected:
    (h, w) = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 2)
    cv2.putText(image, "Fake Note", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Display the image with the result
cv2.imshow("Image", image)
cv2.waitKey(0)
