"""
test_doorbell.py - Test the doorbell system
Run this to simulate a doorbell press and test recognition
"""

from doorbell_system import SmartDoorbellSystem
import time

print("="*60)
print("Smart Doorbell Test")
print("="*60)

# Initialize system
doorbell = SmartDoorbellSystem()

# Check if faces are registered
if len(doorbell.known_faces['names']) == 0:
    print("\n⚠ WARNING: No faces registered!")
    print("Run register_faces.py first!")
    exit(1)

print(f"\nRegistered faces: {', '.join(doorbell.known_faces['names'])}")
print("\n Taking photo")

# Simulate doorbell press
result = doorbell.doorbell_pressed(camera_index=0)

print(f"\n✓ Test complete! Result: {result}")