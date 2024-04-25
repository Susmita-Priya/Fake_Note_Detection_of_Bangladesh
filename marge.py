import cv2
import numpy as np
import imutils

# Load the image and convert it to grayscale
image = cv2.imread("pic/1000tk.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to detect edges
edged = cv2.Canny(blurred, 30, 150)

# Find contours in the edged image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Initialize the flag for fake note detection, UV/IR detection, and watermark detection
fake_note_detected = False
uv_ir_detected = False
watermark_detected = False

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
            # Draw a green rectangle around the contour
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Mark the contour area
            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
            cv2.putText(image, "Real Note", (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (4, 2, 115), 3)
            fake_note_detected = True

    # Apply UV/IR detection based on the intensity range
    mask_uv = cv2.inRange(blurred, 0, 50)  # Adjust the intensity range for UV detection
    mask_ir = cv2.inRange(blurred, 150, 255)  # Adjust the intensity range for IR detection
    uv_pixels = cv2.countNonZero(mask_uv)
    ir_pixels = cv2.countNonZero(mask_ir)

    if uv_pixels > 1000 or ir_pixels > 1000:  # Adjust the threshold for UV/IR detection
        uv_ir_detected = True
        # Mark the UV/IR detected areas
        image[mask_uv > 0] = [0, 0, 255]  # Set the color to red for UV detection
        image[mask_ir > 0] = [0, 255, 0]  # Set the color to green for IR detection

    # Apply watermark detection based on specific features or patterns
    # You will need to define the watermark detection logic here
    # If the watermark is detected, set the watermark_detected flag to True
    # Mark the detected areas by drawing contours or any other visual indication

if not fake_note_detected:
    (h, w) = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 2)
    cv2.putText(image, "Fake Note", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

# Create a single window to display the images
combined = np.hstack([gray, blurred, edged])

# Define the desired fixed height
fixed_height = 300

# Calculate the aspect ratio of the combined image
combined_aspect_ratio = combined.shape[1] / combined.shape[0]

# Calculate the width based on the fixed height
# fixed_width = int(fixed_height * combined_aspect_ratio)
# Set the maximum width for display
max_width = 1200

# Resize the combined image to the fixed width and height
if combined.shape[1] > max_width:
    combined = cv2.resize(combined, (max_width, fixed_height))

# Display the grayscale image, Gaussian blur, and edge detection in a single window
cv2.imshow("Grayscale | Gaussian Blur | Edges ", combined)
cv2.waitKey(0)


image = cv2.resize(image, (700, fixed_height))
# Display the image with the detected contours, UV/IR, and watermark areas
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
