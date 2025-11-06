"""
register_faces.py - Register known faces from face_data folder
Run this after adding images to face_data/person1, person2, person3
"""

from doorbell_system import SmartDoorbellSystem, train_and_register_faces

print("="*60)
print("Registering Known Faces")
print("="*60)

# Initialize the system
print("\nInitializing Smart Doorbell System...")
doorbell = SmartDoorbellSystem(
    model_path='face_recognition_model.tflite',
    embeddings_path='known_faces.pkl',
    threshold=0.6
)

# Register faces from face_data directory
print("\nScanning face_data directory...")
train_and_register_faces(doorbell, data_dir='face_data')

print("\n" + "="*60)
print("✓ Face registration complete!")
print("✓ Embeddings saved to: known_faces.pkl")
print("="*60)