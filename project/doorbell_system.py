# Try to use tflite-runtime (lighter for RPi), fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    USING_TFLITE_RUNTIME = False

import numpy as np
import cv2
from pathlib import Path
import pickle
import time
from datetime import datetime

class SmartDoorbellSystem:
    def __init__(self, model_path='face_recognition_model.tflite', 
                 embeddings_path='known_faces.pkl',
                 input_size=128,
                 threshold=0.6):
        
        self.input_size = input_size
        self.threshold = threshold
        self.embeddings_path = embeddings_path
        
        # Load TFLite model
        if USING_TFLITE_RUNTIME:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load known face embeddings
        self.known_faces = self.load_known_faces()
        
        # Load Haar cascade for face detection
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not Path(cascade_path).exists():
            # Download it if not present
            import urllib.request
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            print("Downloading Haar cascade file...")
            urllib.request.urlretrieve(url, cascade_path)
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def load_known_faces(self):
        # Load known face embeddings from disk
        if Path(self.embeddings_path).exists():
            with open(self.embeddings_path, 'rb') as f:
                return pickle.load(f)
        return {'names': [], 'embeddings': []}
    
    def save_known_faces(self):
        # Save known face embeddings to disk
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.known_faces, f)
    
    def preprocess_face(self, face_img):
        # Preprocess face image for model input
        # Resize to model input size
        face_resized = cv2.resize(face_img, (self.input_size, self.input_size))
        
        # Convert to RGB if needed
        if len(face_resized.shape) == 2:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [-1, 1] (MobileNet preprocessing)
        face_normalized = (face_resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension
        return np.expand_dims(face_normalized, axis=0)
    
    def get_face_embedding(self, face_img):
        # Get embedding for a single face image
        input_data = self.preprocess_face(face_img)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output embedding
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding[0]
    
    def cosine_similarity(self, emb1, emb2):
        # Calculate cosine similarity between two embeddings
        return np.dot(emb1, emb2)
    
    def recognize_face(self, face_img):
        # Recognize a single face image
        if len(self.known_faces['names']) == 0:
            return "Unknown", 0.0
        
        # Get embedding for input face
        embedding = self.get_face_embedding(face_img)
        
        # Compare with all known faces
        best_match = None
        best_similarity = -1
        
        for name, known_embedding in zip(self.known_faces['names'], 
                                         self.known_faces['embeddings']):
            similarity = self.cosine_similarity(embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.threshold:
            return best_match, best_similarity
        else:
            return "Unknown", best_similarity
    
    def register_face(self, face_img, name):
        # Register a new face with given name
        embedding = self.get_face_embedding(face_img)
        
        self.known_faces['names'].append(name)
        self.known_faces['embeddings'].append(embedding)
        
        self.save_known_faces()
        print(f"Registered {name} successfully!")
    
    def detect_and_recognize(self, image):
        # Detect faces in the image and recognize them
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Recognize face
            name, confidence = self.recognize_face(face_roi)
            
            results.append((x, y, w, h, name, confidence))
        
        return results
    
    def doorbell_pressed(self, camera_index=0, save_image=True):
        # Handle doorbell press event
        print(f"\n{'='*50}")
        print(f"Doorbell pressed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")
        
        # Open camera - try different camera indices if specified one fails
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Camera {camera_index} failed, trying alternate cameras...")
            # Try other common camera indices
            for alt_idx in [0, 1, 2]:
                if alt_idx != camera_index:
                    cap = cv2.VideoCapture(alt_idx)
                    if cap.isOpened():
                        print(f"Using camera {alt_idx} instead")
                        camera_index = alt_idx
                        break
        
        if not cap.isOpened():
            print(f"ERROR: Could not open any camera")
            return "Unknown"
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Wait for camera to warm up and stabilize
        print("Warming up camera...")
        time.sleep(1.0)
        
        # Discard first few frames (often blurry)
        for _ in range(3):
            cap.read()
        
        # Capture the actual frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("ERROR: Failed to capture image!")
            return "Unknown"
        
        # Detect and recognize faces
        results = self.detect_and_recognize(frame)
        
        best_result = None
        best_confidence = 0
        
        # Process all detected faces
        for (x, y, w, h, name, confidence) in results:
            # Update best result
            if name != "Unknown" and confidence > best_confidence:
                best_result = name
                best_confidence = confidence
            
            # Draw rectangle and label on image
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save image with timestamp
        if save_image:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"doorbell_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as: {filename}")
        
        # Display image briefly (skip if running headless on RPi)
        try:
            cv2.imshow('Smart Doorbell - Press any key to close', frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
        except:
            # Running headless, just save the image
            print("Running in headless mode (no display)")
        
        # Determine final result
        if best_result and best_confidence >= self.threshold:
            result = f"KNOWN PERSON: {best_result}"
        else:
            result = "UNKNOWN PERSON"
        
        print(f"\n{'='*50}")
        print(f"Result: {result}")
        print(f"Confidence: {best_confidence:.2f}")
        print(f"{'='*50}\n")
        
        return "Known" if best_result else "Unknown"


def create_mobilenet_model(input_size=128, embedding_size=128):
    # Create a MobileNetV2-based model for face embeddings
    if USING_TFLITE_RUNTIME:
        print("ERROR: Cannot create model with tflite-runtime.")
        print("Please run this on a machine with full TensorFlow installed.")
        return None
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.5  # Width multiplier - using 0.5 for efficiency
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom head for embeddings
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(embedding_size, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
    ])
    
    return model


def convert_to_tflite(keras_model, output_path='face_recognition_model.tflite'):
    # Convert Keras model to TensorFlow Lite format
    if USING_TFLITE_RUNTIME:
        print("ERROR: Cannot convert model with tflite-runtime.")
        print("Please run this on a machine with full TensorFlow installed.")
        return
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Optimization for Raspberry Pi
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_path}")


def train_and_register_faces(system, data_dir='face_data'):
    # Train and register faces from the given data directory
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Directory {data_dir} not found!")
        return
    
    for person_dir in data_path.iterdir():
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        print(f"\nRegistering {person_name}...")
        
        # Get all images for this person (support multiple extensions)
        images = (list(person_dir.glob('*.jpg')) + 
                  list(person_dir.glob('*.JPG')) +
                  list(person_dir.glob('*.jpeg')) + 
                  list(person_dir.glob('*.JPEG')) +
                  list(person_dir.glob('*.png')) + 
                  list(person_dir.glob('*.PNG')))
        
        if not images:
            print(f"No images found for {person_name}")
            continue
        
        # Use the first clear image for registration
        registered = False
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Could not read: {img_path.name}")
                continue
            
            # Detect face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = system.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = img[y:y+h, x:x+w]
                system.register_face(face_roi, person_name)
                registered = True
                break
            else:
                print(f"  No face detected in: {img_path.name}")
        
        if not registered:
            print(f"  WARNING: Could not register {person_name} - no valid faces found")
