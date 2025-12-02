from doorbell_system import SmartDoorbellSystem, train_and_register_faces


print("Registering Known Faces")


# Initialize the system
print("\nInitializing Smart Doorbell System")
doorbell = SmartDoorbellSystem(
    model_path='face_recognition_model.tflite',
    embeddings_path='known_faces.pkl',
    threshold=0.6
)

# Register faces from face_data directory
print("\nScanning face_data directory...")
train_and_register_faces(doorbell, data_dir='face_data')


print("Face registration complete!")
print("Embeddings saved to: known_faces.pkl")
