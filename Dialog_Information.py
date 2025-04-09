from PyQt5.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,QLineEdit,QFormLayout
from PyQt5.QtGui import QImage, QPixmap
import cv2 as cv


class RegistrationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Registration information")
        self.name_input = QInputDialog(self)
        self.id_input = QInputDialog(self)
        self.student_name, ok1 = QInputDialog.getText(self, "Enter name", "Name:")
        self.student_id, ok2 = QInputDialog.getText(self, "ID", "Student ID:")

    def get_inputs(self):
        return self.student_name.strip(), self.student_id.strip()

class ConfirmationDialog(QDialog):
    def __init__(self, student_id, name, face_img):
        super().__init__()
        self.setWindowTitle("Confirm")
        self.result = False

        layout = QVBoxLayout()
        face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        height, width, channel = face_img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(face_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)
        info_label = QLabel(f"Student ID: {student_id}\nName: {name}")
        layout.addWidget(info_label)
        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.confirm_button.clicked.connect(self.confirm)
        self.cancel_button.clicked.connect(self.cancel)

    def confirm(self):
        self.result = True
        self.accept()

    def cancel(self):
        self.reject()
