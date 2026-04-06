import cv2
import dlib
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
FACE_ENCODER_PATH = BASE_DIR / "models" / "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
IMAGES_DIR = BASE_DIR / "images"

face_detector = dlib.get_frontal_face_detector() #type: ignore
face_encoder = dlib.face_recognition_model_v1(str(FACE_ENCODER_PATH)) #type: ignore
shape_predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH)) #type: ignore

def load_people_images(base_dir):
    people = {}
    for person_name in os.listdir(base_dir):
        folder = os.path.join(base_dir, person_name)
        if os.path.isdir(folder):
            image_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            if image_paths:
                people[person_name] = image_paths
    return people

def get_face_descriptor(image_paths):
    descriptors = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        faces = face_detector(img, 1)
        if faces:
            shape = shape_predictor(img, faces[0])
            descriptor = np.array(face_encoder.compute_face_descriptor(img, shape))
            descriptors.append(descriptor)
    return np.mean(descriptors, axis=0) if descriptors else None

def preload_known_faces():
    print("Loading known faces...")
    people = load_people_images(IMAGES_DIR)
    known_faces = {}
    for name, paths in people.items():
        desc = get_face_descriptor(paths)
        if desc is not None:
            known_faces[name] = desc
            print(f"  ✅ Loaded {name}")
    print(f"✅ Total known faces: {len(known_faces)}")
    return known_faces

known_faces = preload_known_faces()

def process_identity_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    identified_person = "Unknown"

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        shape = shape_predictor(frame, dlib.rectangle(x, y, x + w, y + h)) #type: ignore
        descriptor = np.array(face_encoder.compute_face_descriptor(frame, shape))

        identity, min_distance = "Unknown", 0.4

        for name, stored_desc in known_faces.items():
            distance = np.linalg.norm(stored_desc - descriptor)
            if distance < min_distance:
                min_distance = distance
                identity = name

        identified_person = identity

    return identified_person