import sqlite3
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F
import numpy as np
from config import *
import cv2 as cv
import json


def eye_aspect_ratio(landmarks, indices):
    eye_points = np.array([landmarks[i] for i in indices])
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)
def enhance_image_if_needed(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    brightness = np.mean(y_channel)
    print("Brightness : {:.2f}".format(brightness))
    if brightness < BRIGHTNESS_THRESHOLD :
        ycrcb[:, :, 0] = cv.equalizeHist(y_channel)
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
        print("The image is too dark -> Has increased brightness")
    return image

def cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()


def crop_face_from_bbox(image, bbox, margin=15, image_size=160):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, w)
    y2 = min(y2 + margin, h)
    face_crop = image[y1:y2, x1:x2]
    face_crop_ = enhance_image_if_needed(face_crop)
    face_crop_ = cv.resize(face_crop_, (image_size, image_size))
    transform = Compose([
        ToTensor(),
        Normalize([0.5], [0.5])
    ])
    return face_crop, transform(face_crop_).unsqueeze(0)


def estimate_head_pose(landmarks, frame_shape):
    landmarks = np.array(landmarks, dtype="double")
    image_points = landmarks[[30, 8, 36, 45, 48, 54], :2]

    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv.solvePnP(
        objectPoints=np.array([
            (0.0, 0.0, 0.0),          # nose
            (0.0, -63.6, -12.5),      # chin
            (-43.3, 32.7, -26.0),     # left eye corner
            (43.3, 32.7, -26.0),      # right eye corner
            (-28.9, -28.9, -24.1),    # left corner of mouth
            (28.9, -28.9, -24.1)      # right corner of mouth
        ], dtype="double"),
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(proj_matrix)

    pitch, yaw, roll = euler_angles.flatten()
    if pitch > 90:
        pitch -= 180
    if yaw > 90:
        yaw -= 180
    if roll > 90:
        roll -= 180
    return pitch, yaw, roll

def detect_face_with_pose(model_face, frame) :
    faces = model_face.get(frame)
    if not faces :
        return None, None, None
    main_face = max(faces, key=lambda face : face.bbox[2] - face.bbox[0])
    bbox = list(map(int, main_face.bbox))
    landmarks = main_face.kps
    pose = main_face.pose
    return bbox, landmarks, pose

def is_face_image_good(image, landmarks, frame_shape):
    pitch, yaw, roll = estimate_head_pose(landmarks, frame_shape)
    if pitch is None or yaw is None:
        return False, "Can't get head pose"

    if abs(pitch) > MAX_PITCH or abs(yaw) > MAX_YAW:
        return False, "The head rotates to much (pitch={:.2f}, yaw={:.2f})".format(pitch, yaw)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
    if laplacian_var < 100:
        return False, "The image is too blurry (Sharpness: {:.2f})".format(laplacian_var)

    brightness = gray.mean()
    if brightness < 70:
        return False, "The image is too dark (Brightness: {:.2f})".format(brightness)

    return True, "Quality Image"

def is_face_in_ellipe(face_bbox, ellipse_center, axes) :
    x1, y1, x2, y2 = face_bbox
    face_center = ((x1 + x2)/2, (y1+y2)/2)
    dx = face_center[0] - ellipse_center[0]
    dy = face_center[1] - (ellipse_center[1] + 10)
    a, b = axes
    value = (dx**2) /(a**2) + (dy**2) /(b**2)
    return value<= 1


def find_best_match(embedding_cam, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    best_match = None
    best_score = -1
    try:
        cursor.execute("SELECT student_id, name, embedding FROM faces")
        rows = cursor.fetchall()

        for student_id, name, embedding_str in rows:
            embedding_reg = torch.tensor(json.loads(embedding_str), dtype=torch.float32)
            embedding_reg = embedding_reg.to(embedding_cam.device)
            score = cosine_similarity(embedding_cam, embedding_reg)
            if score > best_score:
                best_score = score
                best_match = {
                    'student_id': student_id,
                    'name': name,
                    'score': score
                }

        if best_score >= SIMILARITY_THRESHOLD:
            return best_match
        else:
            return None

    except sqlite3.Error as e:
        print("❌ Error access database:", e)
        return None

    finally:
        conn.close()
