import cv2
import os

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

DATA_PATH = "face_data"   # main folder for all faces

# Ask for student roll no / name
student_id = input("Enter student roll no or name: ")

# Create folder for that student automatically
folder_path = os.path.join(DATA_PATH, student_id)
os.makedirs(folder_path, exist_ok=True)

cap = cv2.VideoCapture(0)   # open webcam
count = 0
max_images = 20             # number of photos to save

print("\n📸 Look at the camera — move slightly left/right/up/down.")
print("Press 'q' to stop early.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # crop face in color
        count += 1
        filename = f"{student_id}_{count}.jpg"
        cv2.imwrite(os.path.join(folder_path, filename), face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Saved {count}/{max_images}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        print(f"✅ Saved {filename}")

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= max_images:
        print("\n✅ Dataset collection complete.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nAll images for '{student_id}' are saved in '{folder_path}'.")
