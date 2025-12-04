import time
import RPi.GPIO as GPIO 
from datetime import datetime

BUTTON_PIN = 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# capturing image first 
print("\nCapturing image NOW...")
start_time = time.time()

import cv2

# Open camera failsafe
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera 0 failed, trying alternates...")
    for alt_idx in [1, 2]:
        cap = cv2.VideoCapture(alt_idx)
        if cap.isOpened():
            print(f"Using camera {alt_idx}")
            break

if not cap.isOpened():
    print("ERROR: Could not open any camera")
    exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Camera warm-up, frame discard and capture

time.sleep(1.0)

for _ in range(15):
    cap.read()

while True:
    if GPIO.input(BUTTON_PIN) == GPIO.LOW:
        print("Button Pressed")
        main_start_time = time.time()
        break

ret, frame = cap.read()
cap.release()
GPIO.cleanup()
capture_end_time = time.time()

if ret == True:
    print("Captured Image") 
else:
    print("ERROR: Failed to capture image!")
    exit(1)

# saving raw image
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
capture_filename = f"capture_{timestamp}.jpg"
cv2.imwrite(capture_filename, frame)
print(f"Saved raw capture as: {capture_filename}")

# inference
print("\nLoading recognition system...")
inference_start = time.time()

# Import heavy modules AFTER capture
from doorbell_system import SmartDoorbellSystem

# Initialize system
doorbell = SmartDoorbellSystem()

# Check if faces registered
if len(doorbell.known_faces['names']) == 0:
    print("No faces registered, run register_faces.py first")
    exit(1)

# Detect and recognize on captured frame
results = doorbell.detect_and_recognize(frame)

best_result = None
best_confidence = 0

for (x, y, w, h, name, confidence) in results:
    if name != "Unknown" and confidence > best_confidence:
        best_result = name
        best_confidence = confidence
    
    # Draw on image
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    label = f"{name} ({confidence:.2f})"
    cv2.putText(frame, label, (x, y-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Save annotated image
annotated_filename = f"doorbell_{timestamp}.jpg"
cv2.imwrite(annotated_filename, frame)


# Results
if best_result and best_confidence >= doorbell.threshold:
    result = f"KNOWN PERSON: {best_result}"
    final_result = "Known"
else:
    result = "UNKNOWN PERSON"
    final_result = "Unknown"

inference_end_time = time.time()  # ADD THIS LINE

# Calculate timings
capture_time = capture_end_time - main_start_time
inference_time = inference_end_time - inference_start
total_time = inference_end_time - main_start_time


print(f"Result: {result}")
print(f"Confidence: {best_confidence:.2f}")
print(f"")
print(f"Timing Breakdown:")
print(f"  Image Capture: {capture_time:.2f}s")
print(f"  Face Recognition: {inference_time:.2f}s")
print(f"  Total Time: {total_time:.2f}s")
print(f"")
print(f"Images saved:")
print(f"  Raw: {capture_filename}")
print(f"  Annotated: {annotated_filename}")