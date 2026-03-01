import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import smtplib
import ssl
from email.message import EmailMessage
import os
from datetime import datetime

# ------------------------------
# EMAIL CONFIGURATION
# ------------------------------
SENDER_EMAIL = "poertypal@gmail.com" # your email
SENDER_PASSWORD = "mwkeqcrrjmepddam"  # App password, not your actual Gmail password
RECEIVER_EMAIL = "amruthaa_tirumalasetty@srmap.edu.in"

def send_email_alert(person_name, image_path):
    msg = EmailMessage()
    msg["Subject"] = f"⚠️ Mobile Phone Detected - {person_name}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg.set_content(f"""
    Alert generated on: {time_now}
    Person Detected: {person_name}
    Event: Mobile phone usage detected in front of camera.
    Please find the attached image for reference.
    """)

    # Attach the snapshot
    with open(image_path, "rb") as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    # Send the email securely
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print(f"📧 Alert email sent to {RECEIVER_EMAIL} for {person_name}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

# ------------------------------
# Load face encodings
# ------------------------------
known_face_encodings = np.load("encodings.npy", allow_pickle=True)
known_face_names = np.load("names.npy", allow_pickle=True)
print(f"✅ Loaded {len(known_face_names)} known face encodings.")

# ------------------------------
# YOLOv8 Model
# ------------------------------
yolo_model = YOLO("yolov8n.pt")

# ------------------------------
# Create alert folder
# ------------------------------
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# ------------------------------
# Video Capture
# ------------------------------
cap = cv2.VideoCapture(0)
alert_cooldown = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    # YOLO phone detection
    results = yolo_model(frame)[0]
    phone_detected = False

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        label = yolo_model.names[cls_id]

        if label == "cell phone" and conf > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # If phone detected and not recently alerted
        if phone_detected:
            now = datetime.now()
            if name not in alert_cooldown or (now - alert_cooldown[name]).seconds > 60:
                img_path = f"alerts/{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(img_path, frame)
                send_email_alert(name, img_path)
                alert_cooldown[name] = now

    # Display
    cv2.imshow("Face + Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting program.")
