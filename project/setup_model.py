from doorbell_system import create_mobilenet_model, convert_to_tflite

print("Creating MobileNet Face Recognition Model")


# Create the model
print("\n1. Creating MobileNetV2 model")
model = create_mobilenet_model(input_size=128, embedding_size=128)

print("\n2. Converting to TensorFlow Lite")
convert_to_tflite(model, output_path='face_recognition_model.tflite')

print("Model created successfully!")
print("Saved as: face_recognition_model.tflite")
