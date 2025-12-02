# REDUNDANT FILE - IGNORE


from doorbell_system import SmartDoorbellSystem
import time

print("="*60)
print("Smart Doorbell Test")
print("="*60)

# Initialize system
print("\nInitializing Smart Doorbell System...")
doorbell = SmartDoorbellSystem()

# Check if faces are registered
if len(doorbell.known_faces['names']) == 0:
    print("\n⚠ WARNING: No faces registered!")
    print("Run register_faces.py first!")
    exit(1)

print(f"\nRegistered faces: {', '.join(doorbell.known_faces['names'])}")

# CHANGE THIS NUMBER based on your camera test above
# Try 0, 1, 2, 3, etc. until you find your external camera
result = doorbell.doorbell_pressed(camera_index=2)  # <-- Change this number

print(f"\n✓ Test complete! Result: {result}")
print("Check the saved doorbell_TIMESTAMP.jpg image!")
print("="*60)