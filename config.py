from facenet_pytorch import InceptionResnetV1
import torch
import insightface

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

facenet_model = InceptionResnetV1(pretrained='vggface2', device=device).eval()

model_face = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model_face.prepare(ctx_id=0)
# Chỉ số landmark mắt (theo dlib/68-point)
left_eye_indices = [36, 37, 38, 39, 40, 41]
right_eye_indices = [42, 43, 44, 45, 46, 47]

SIMILARITY_THRESHOLD = 0.85
EYE_THRESHOLD = 0.1
CONSECUTIVE_FRAMES = 2
COS_THRESHOLD = 0.8
MAX_YAW = 15   # giới hạn góc quay đầu (độ)
MAX_PITCH = 15
BRIGHTNESS_THRESHOLD = 70
MIN_CLARITY = 20

FACE_DATA_FILE = 'face_data.json'
DB_PATH = 'face_db.db'