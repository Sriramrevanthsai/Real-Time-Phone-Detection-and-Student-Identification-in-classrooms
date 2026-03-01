import os
import cv2
import numpy as np
import face_recognition
from sympy import python

# ------------------------------
# Paths
# ------------------------------
dataset_dir = "face_data"  # your folder name
encodings_file = "encodings.npy"
names_file = "names.npy"

# Create lists to store encodings and names
known_face_encodings = []
known_face_names = []

# ------------------------------
# Loop through each person in dataset
# ------------------------------
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read image {image_path}, skipping...")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces (use "hog" for Windows stability)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if len(face_locations) == 0:
            print(f"⚠️ No face found in {image_path}, skipping...")
            continue

        # Get encodings
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        for encoding in encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)

        print(f"✅ Processed {image_path}")

# ------------------------------
# Save encodings and names
# ------------------------------
np.save(encodings_file, known_face_encodings)
np.save(names_file, known_face_names)
print(f"🎉 Saved {len(known_face_names)} face encodings to '{encodings_file}' and '{names_file}'")
