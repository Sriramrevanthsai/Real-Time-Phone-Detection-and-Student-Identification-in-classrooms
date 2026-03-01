# Real-Time-Phone-Detection-and-Student-Identification-in-classrooms

📌 Overview

This project presents a real-time classroom monitoring system that detects mobile phone usage and identifies the student involved using deep learning and face recognition techniques.

The system integrates:

YOLOv8 for mobile phone detection

Face Recognition (128D embeddings) for student identification

Automated email alert system for instructor notification

The goal is to reduce classroom distractions and automate monitoring using machine learning.

🎯 Key Features

Real-time mobile phone detection using YOLOv8n

Student identification using face embeddings

128-dimensional facial encoding system

Automated email alert with timestamp and evidence snapshot

Cooldown mechanism to prevent repeated alerts

End-to-end ML pipeline implementation

🧠 System Architecture

The system consists of three major modules:

1️⃣ Data Acquisition Module

Captures student face images during dataset creation

Captures live webcam frames during monitoring

2️⃣ Processing Module

YOLOv8n object detection for mobile phone detection

Face encoding comparison using Euclidean distance

Mapping detected face to student identity

3️⃣ Alert & Storage Module

Saves timestamped evidence images

Sends automated email alerts via SMTP

Prevents duplicate alerts using cooldown logic

⚙️ Methodology
Step 1: Dataset Collection

Minimum 20 face images per student

Variations in angle, lighting, and facial expression

Organized into student-specific folders

Step 2: Preprocessing

RGB conversion

Haar Cascade face detection

Cropping and resizing

Normalization

Step 3: Face Encoding

Used face_recognition library

Generated 128-dimensional embeddings

Stored in:

encodings.npy

names.npy

Step 4: Mobile Phone Detection

Integrated Ultralytics YOLOv8n

Confidence threshold > 0.70

Real-time inference

Step 5: Real-Time Recognition

Detect face in frame

Generate embedding

Compare with stored embeddings

Identify student using Euclidean distance

Step 6: Alert Trigger

When:

Phone is detected
AND

Student face is recognized

Then:

Snapshot is saved

Email alert is sent to instructor

Cooldown timer prevents repeated alerts

📊 Performance & Results

Stable frame rate: 15–20 FPS

High-confidence phone detection (> 0.70)

Strong face recognition accuracy under normal lighting

Reliable automated email alert system

Minimal inference delay

The system successfully achieved all project objectives and demonstrated real-time classroom monitoring capability.

🛠️ Technologies Used

Python 3.10

OpenCV

Ultralytics YOLOv8

face_recognition (Dlib-based 128D embeddings)

NumPy

SMTP (Email alerts)

📂 Project Files

create_a_dataset.py → Capture student face images

train_faces.py → Generate and store face embeddings

main.py → Real-time detection + recognition + alert system

🚧 Limitations

Reduced performance in low-light conditions

Face recognition issues with partial occlusion

Possible YOLO false positives (similar objects)

Webcam quality impacts accuracy

Internet required for email alerts

🚀 Future Scope

Live monitoring dashboard

Detection log database storage

Attendance tracking

Head pose & attention monitoring

Custom YOLO training for classroom-specific dataset

Mobile push notification alerts

🎓 Conclusion

This project demonstrates the practical integration of deep learning, object detection, and face recognition into a real-time monitoring system. It highlights the real-world application of supervised learning, CNN-based detection models, feature extraction, and inference pipelines.

The system is scalable, practical, and aligned with modern AI-driven educational technologies.
