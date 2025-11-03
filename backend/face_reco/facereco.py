import cv2
import dlib
import numpy as np
import os

# ✅ Load heavy models ONCE (on module import)
face_detector = dlib.get_frontal_face_detector() #type: ignore
face_encoder = dlib.face_recognition_model_v1( "/Users/ishan/dev/kaira_software/backend/face_reco/models/dlib_face_recognition_resnet_model_v1.dat")   #type: ignore
shape_predictor = dlib.shape_predictor("/Users/ishan/dev/kaira_software/backend/face_reco/models/shape_predictor_68_face_landmarks.dat") #type: ignore

# Base directory for known faces
base_dir = "/Users/ishan/dev/kaira_software/backend/face_reco/images"


def load_people_images(base_dir):
    """Return dict of person_name -> list of image paths."""
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
    """Compute average descriptor for a list of face images."""
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


# ✅ Preload known faces ONCE
def preload_known_faces():
    print("Loading known faces...")
    people = load_people_images(base_dir)
    known_faces = {}
    for name, paths in people.items():
        desc = get_face_descriptor(paths)
        if desc is not None:
            known_faces[name] = desc
            print(f"  ✅ Loaded {name}")
    print(f"✅ Total known faces: {len(known_faces)}")
    return known_faces


# Keep global cache
known_faces = preload_known_faces()


def process_identity_from_frame(frame):
    """Recognize identity from frame using preloaded models and faces."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    identified_person = "Unknown"

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        shape = shape_predictor(frame, dlib.rectangle(x, y, x + w, y + h)) #type: ignore
        descriptor = np.array(face_encoder.compute_face_descriptor(frame, shape))

        identity, min_distance = "Unknown", 0.4  # threshold

        for name, stored_desc in known_faces.items():
            distance = np.linalg.norm(stored_desc - descriptor)
            if distance < min_distance:
                min_distance = distance
                identity = name

        identified_person = identity

    return identified_person
