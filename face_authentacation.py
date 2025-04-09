import sys
import time
import warnings
from utils import *
from database import *
from PyQt5.QtWidgets import QApplication

warnings.filterwarnings("ignore", category=FutureWarning)



def main() :
    init_db()
    cap = cv.VideoCapture(0)
    if not cap.isOpened() :
        print("Can't open camera!")
        return

    try_count = 0
    ok = False
    face_tensor_ = None
    ellipse_axes_big = (180, 250)
    ellipse_axes_small = (30, 40)
    ellipse_center = (320, 240)
    while try_count < 3 and cap.isOpened() and not ok :
        print("Try time {}:".format(try_count + 1))
        trial_start = time.time()
        ok = False
        while time.time() - trial_start < 8 :
            ret, frame = cap.read()
            if not ret :
                break
            frame = cv.flip(frame, 1)
            height, width, _ = frame.shape
            remaining = int(3 - (time.time() - trial_start)) + 1
            cv.putText(frame, str(remaining), (50, 50),
                cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv.ellipse(mask, ellipse_center, ellipse_axes_big, 0, 0, 360, (255, 255, 255), -1)
            white_bg = np.ones_like(frame, dtype=np.uint8) * 255
            frame_focus = cv.bitwise_and(frame, mask)
            frame_outside = cv.bitwise_and(white_bg, cv.bitwise_not(mask))
            frame = cv.add(frame_focus, frame_outside)
            cv.ellipse(frame, ellipse_center, ellipse_axes_big, 0, 0, 360, (0, 255, 255), 2)
            if time.time() - trial_start >= 3 :
                faces = model_face.get(frame)
                for face in faces :
                    landmarks = face.landmark_3d_68
                    bbox = face.bbox
                    bbox_int = list(map(int, bbox))
                    if not is_face_in_ellipe(bbox_int,axes=ellipse_axes_small, ellipse_center=ellipse_center) :
                        cv.putText(frame, "Put face in picture please !", (100, height - 50),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue
                    else:
                        cv.putText(frame, "Keep your face still please !", (100, height - 50),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print("Face in frame")

                        is_good, message = is_face_image_good(frame, landmarks, frame.shape)
                        if not is_good :
                            print("❌ Face not passed:", message)
                            continue
                        else :
                            print("Face passed:", message)
                            ok = True
                        face_crop ,face_tensor_ = crop_face_from_bbox(frame, bbox)
                        face_tensor_ = face_tensor_.to(device)
                        break
            cv.imshow('Camera', frame)
            if cv.waitKey(1) & 0xFF == ord(" ") :
                cap.release()
                cv.destroyAllWindows()
                return
            if ok :
                break
        if face_tensor_ is not None :
            with torch.no_grad() :
                embedding = facenet_model(face_tensor_)
            result = find_best_match(embedding.squeeze(0))
            if result :
                student_id = result['student_id']
                name = result['name']
                score = result['score']
                print("Name : {}, ID : {}, Score : {}, ".format(student_id, name, score))

        if not ok :
            print("❌ Failed on the 1st/3rd attempt.  {}.".format(try_count + 1))
        try_count += 1
        time.sleep(1)

    cap.release()
    cv.destroyAllWindows()

    if not ok :
        print("❌ Face failed in 3 attempts.")
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()