"""
quick_doorbell.py - Ultra-fast image capture + recognition
Captures image IMMEDIATELY using doorbell_system method
Optimized for instant response when doorbell is pressed
"""

import time
from datetime import datetime

print("="*60)
print("Quick Doorbell - Instant Capture")
print("="*60)

# STEP 1: INSTANT CAPTURE (happens immediately)
print("\nCapturing image NOW...")
start_time = time.time()

# Import only what we need for capture
import cv2

# Open camera - auto-detect if camera 0 fails
cap = cv2.VideoCapture(2)

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Camera warm-up
time.sleep(1.0)

# Discard first frames
for _ in range(15):
    cap.read()

# CAPTURE THE FRAME
ret, frame = cap.read()
cap.release()

capture_time = time.time() - start_time
print(f"✓ Image captured in {capture_time:.2f} seconds")

if not ret or frame is None:
    print("ERROR: Failed to capture image!")
    exit(1)

# Save the raw capture
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
capture_filename = f"capture_{timestamp}.jpg"
cv2.imwrite(capture_filename, frame)
print(f"✓ Saved raw capture as: {capture_filename}")

# STEP 2: NOW DO THE INFERENCE (slower part)
print("\nLoading recognition system...")
inference_start = time.time()

# Import heavy modules AFTER capture
from doorbell_system import SmartDoorbellSystem

# Initialize system
doorbell = SmartDoorbellSystem()

# Check if faces registered
if len(doorbell.known_faces['names']) == 0:
    print("⚠ WARNING: No faces registered!")
    print("Run register_faces.py first!")
    exit(1)

print(f"Registered faces: {', '.join(doorbell.known_faces['names'])}")
print("Running face recognition on captured image...")

# Detect and recognize on the ALREADY CAPTURED frame
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

inference_time = time.time() - inference_start
total_time = time.time() - start_time

# Results
if best_result and best_confidence >= doorbell.threshold:
    result = f"KNOWN PERSON: {best_result}"
    final_result = "Known"
else:
    result = "UNKNOWN PERSON"
    final_result = "Unknown"

print("\n" + "="*60)
print(f"Result: {result}")
print(f"Confidence: {best_confidence:.2f}")
print(f"")
print(f"Timing Breakdown:")
print(f"  Capture time: {capture_time:.2f}s")
print(f"  Inference time: {inference_time:.2f}s")
print(f"  Total time: {total_time:.2f}s")
print(f"")
print(f"Images saved:")
print(f"  Raw: {capture_filename}")
print(f"  Annotated: {annotated_filename}")
print("="*60)