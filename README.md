7. Smart Doorbell using Raspberry Pi and ML
 
Motivation: 
    Edge devices like Raspberry Pi are increasingly used for deploying ML systems in real-world settings. Because of this we want to gain experience training and deploying ML models on low level, resource constrained hardware. Creating a smart doorbell with people recognition gives us the opportunity to build a useful real world device while learning about ML deployment.
   
Design Goals: 
    Use Raspberry Pi with a camera module to design a smart doorbell that can detect people and classify whether they are recognized or unknown.
    Learn how to deploy ML models on Raspberry Pi
   
Deliverables:
    Implement a lightweight face detection or person detection model using TensorFlow Lite.
    Build a simple system that, when someone appears at the door, captures an image and classify
    Optional: Add an alert mechanism (e.g., send a notification to a phone or log it in a file)
    
Hardware/Software: 
  Needed Hardware:  
    Raspberry with its peripherals such as Power Supply, MicroSD card, Optional: Buzzer for local sound alerts and button for camera activation
    Camera 
   Needed Software: 
    Google Colab 
    CNN based face recognition model 
    Python 
    Libraries such as Tensorflow, OpenCV for Image processing, NumPy
    
Team member responsibilities:
    Research: Everyone 
    Model Training: Joe Paola 
    Hardware setup: Gaurav Hareesh  
    Software development: Everyone
    Writing and Verification: Divyesh

Timeline:
  Week 1-2: Research & Setup
    Gain better understanding of what we need to do to implement the project and gather the materials needed to complete it 
  Week 2-4: Model Training & Optimization
    Train the model to identify known people, convert to tensorflow format
  Week 5-6: Hardware Integration & Programming 
    Put the model onto our system and implement/test the software 

References:
    A lightweight CNN paper on edge vision (https://arxiv.org/abs/1704.04861).
