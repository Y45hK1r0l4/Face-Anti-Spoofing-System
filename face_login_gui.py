import sys
import os
import math
import time
import cv2
import numpy as np
import pickle
import traceback
from ultralytics import YOLO
from cvzone.FaceDetectionModule import FaceDetector
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QMessageBox,
    QFrame,
    QListWidget,
    QDialog,
    QScrollArea,
    QComboBox,
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread

# Constants
USER_DATA_DIR = "user_data"
FACE_ENCODINGS_FILE = os.path.join(USER_DATA_DIR, "face_encodings.pkl")
USER_CREDENTIALS_FILE = os.path.join(USER_DATA_DIR, "user_credentials.pkl")
ATTENDANCE_LOG_DIR = os.path.join(USER_DATA_DIR, "attendance_logs")

# Create directories if they don't exist
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)
if not os.path.exists(ATTENDANCE_LOG_DIR):
    os.makedirs(ATTENDANCE_LOG_DIR)

# YOLO model and face detector initialization
try:
    # Load YOLO model for anti-spoofing
    model = YOLO("models/best_yolov8n.pt")
    classNames = ["fake", "real"]
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    traceback.print_exc()
    model = None
    classNames = ["fake", "real"]

# Initialize the face detector
detector = None


def initialize_face_detector():
    try:
        # minDetectionCon: lower values detect more faces but might have false positives
        # modelSelection: 0 for close-up faces, 1 for longer-distance detection
        return FaceDetector(minDetectionCon=0.5, modelSelection=0)
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        traceback.print_exc()
        return None


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    camera_status_signal = pyqtSignal(bool, str)  # Added camera status signal

    def __init__(self, camera_id=0):
        super().__init__()
        self.running = True
        self.cap = None
        self.camera_id = camera_id
        print(f"VideoThread initialized for camera {camera_id}")

    def run(self):
        try:
            # Capture from specified camera
            print(f"Attempting to open camera {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                error_msg = f"Could not open camera device {self.camera_id}"
                print(error_msg)
                self.error_signal.emit(error_msg)
                self.camera_status_signal.emit(False, error_msg)
                return

            # Set camera resolution
            self.cap.set(3, 640)  # Width
            self.cap.set(4, 480)  # Height

            # Emit success signal
            self.camera_status_signal.emit(
                True, f"Camera {self.camera_id} opened successfully"
            )
            print(f"Camera {self.camera_id} opened successfully")

            # Main frame capture loop
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.change_pixmap_signal.emit(frame)
                else:
                    error_msg = (
                        f"Failed to get frame from camera {self.camera_id}"
                    )
                    print(error_msg)
                    self.error_signal.emit(error_msg)
                    self.camera_status_signal.emit(False, error_msg)
                    break

                # Sleep briefly to avoid high CPU usage
                self.msleep(30)  # 30ms delay (~33 fps)

        except Exception as e:
            error_msg = f"Video thread error: {str(e)}"
            print(error_msg)
            self.error_signal.emit(error_msg)
            self.camera_status_signal.emit(False, error_msg)
            traceback.print_exc()
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                print(f"Camera {self.camera_id} released in run method")

    def stop(self):
        print(f"Stopping video thread for camera {self.camera_id}")
        self.running = False
        self.wait(1000)  # Wait up to 1 second for thread to finish
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print(f"Camera {self.camera_id} released in stop method")
        print(f"Video thread stopped for camera {self.camera_id}")


# Rest of the classes remain unchanged
class AttendanceSystem:
    def __init__(self):
        # Load user data if it exists
        self.known_face_features = []  # Store face features instead of encodings
        self.known_face_names = []
        self.user_credentials = {}
        self.attendance_records = {}  # Store attendance records by date and username

        self.load_data()
        self.load_attendance_records()

    def load_data(self):
        try:
            # Load face features
            if os.path.exists(FACE_ENCODINGS_FILE):
                with open(FACE_ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.known_face_features = data.get("features", [])
                    self.known_face_names = data.get("names", [])

            # Load user credentials
            if os.path.exists(USER_CREDENTIALS_FILE):
                with open(USER_CREDENTIALS_FILE, "rb") as f:
                    self.user_credentials = pickle.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()

    def load_attendance_records(self):
        try:
            # Get current date for attendance log
            current_date = time.strftime("%Y-%m-%d")
            attendance_file = os.path.join(
                ATTENDANCE_LOG_DIR, f"{current_date}.pkl"
            )

            if os.path.exists(attendance_file):
                with open(attendance_file, "rb") as f:
                    self.attendance_records = pickle.load(f)
            else:
                self.attendance_records = {}
        except Exception as e:
            print(f"Error loading attendance records: {e}")
            traceback.print_exc()

    def save_data(self):
        try:
            # Save face features
            with open(FACE_ENCODINGS_FILE, "wb") as f:
                pickle.dump(
                    {
                        "features": self.known_face_features,
                        "names": self.known_face_names,
                    },
                    f,
                )

            # Save user credentials
            with open(USER_CREDENTIALS_FILE, "wb") as f:
                pickle.dump(self.user_credentials, f)
        except Exception as e:
            print(f"Error saving data: {e}")
            traceback.print_exc()

    def save_attendance_record(self, username):
        try:
            # Get current date for attendance log
            current_date = time.strftime("%Y-%m-%d")
            current_time = time.strftime("%H:%M:%S")

            attendance_file = os.path.join(
                ATTENDANCE_LOG_DIR, f"{current_date}.pkl"
            )

            # Update attendance records
            if current_date not in self.attendance_records:
                self.attendance_records[current_date] = {}

            self.attendance_records[current_date][username] = current_time

            # Save to file
            with open(attendance_file, "wb") as f:
                pickle.dump(self.attendance_records, f)

            return current_time
        except Exception as e:
            print(f"Error saving attendance record: {e}")
            traceback.print_exc()
            return None

    def register_user(self, username, password, face_feature):
        # Check if username already exists
        if username in self.user_credentials:
            return False, "Username already exists"

        # Add user credentials
        self.user_credentials[username] = password

        # Add face feature
        self.known_face_features.append(face_feature)
        self.known_face_names.append(username)

        # Save data
        self.save_data()
        return True, "Registration successful"

    def authenticate_user(self, face_feature):
        if not self.known_face_features:
            return False, "No registered users", 0, None

        try:
            # Face matching - find the closest match among stored faces
            best_match_index = -1
            best_match_distance = float("inf")
            confidence = 0

            # Compare with stored face features
            for i, known_feature in enumerate(self.known_face_features):
                # Simple Euclidean distance between feature vectors
                distance = np.linalg.norm(
                    np.array(face_feature) - np.array(known_feature)
                )

                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_index = i

            # Calculate confidence (inverse of distance, normalized)
            # Lower distance = higher confidence
            # Threshold for face matching - may need to adjust based on testing
            match_threshold = 80  # Threshold for considering a match
            if best_match_index >= 0 and best_match_distance < match_threshold:
                # Calculate confidence percentage (100% when distance=0, decreasing as distance increases)
                confidence = max(
                    0, 100 - (best_match_distance / match_threshold * 100)
                )
                # Record attendance for the recognized user
                username = self.known_face_names[best_match_index]
                timestamp = self.save_attendance_record(username)
                return True, username, confidence, timestamp

        except Exception as e:
            print(f"Error in face authentication: {e}")
            traceback.print_exc()
            return False, "Error in face recognition", 0, None

        return False, "Face not recognized", 0, None

    def verify_credentials(self, username, password):
        if (
                username in self.user_credentials
                and self.user_credentials[username] == password
        ):
            return True
        return False

    def get_registered_users(self):
        return self.known_face_names

    def get_todays_attendance(self):
        current_date = time.strftime("%Y-%m-%d")
        if current_date in self.attendance_records:
            return self.attendance_records[current_date]
        return {}


class FaceDetection:
    @staticmethod
    def detect_face_and_antispoofing(frame):
        global detector
        global model

        if detector is None:
            detector = initialize_face_detector()
            if detector is None:
                return None, None, False

        try:
            # Face detection using cvzone's FaceDetector
            img, bboxs = detector.findFaces(frame, draw=False)

            # If face found, check if it's too close
            is_real_face = False
            is_too_close = False

            if bboxs:
                # Get bbox info for the first face detected
                bbox = bboxs[0]

                # Extract the face width and height
                x, y, w, h = bbox["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Check if face is too close to camera
                # Consider face too close if it takes up more than 60% of frame width or height
                frame_height, frame_width = frame.shape[:2]
                width_ratio = w / frame_width
                height_ratio = h / frame_height

                # If face width is more than 60% of frame width or height is more than 60% of frame height
                is_too_close = (width_ratio > 0.5) or (height_ratio > 0.5)

                # Print debug info
                print(
                    f"Face size check - Width ratio: {width_ratio:.2f}, Height ratio: {height_ratio:.2f}, Too close: {is_too_close}")

            # Anti-spoofing using YOLO (only if face is not too close)
            try:
                if model is not None and bboxs and not is_too_close:
                    results = model(frame, stream=True, verbose=False)

                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            cls = int(box.cls[0])

                            if conf > 0.6 and classNames[cls] == "real":
                                is_real_face = True
                                break
                else:
                    # If model is not available or face is too close
                    is_real_face = not is_too_close  # Mark as fake if too close
            except Exception as e:
                print(f"Error in anti-spoofing detection: {e}")
                traceback.print_exc()
                # If anti-spoofing fails, assume real face for debugging (but respect the too close check)
                is_real_face = not is_too_close

            # If face found, extract features and return
            if bboxs:
                # Get bbox info for the first face detected
                bbox = bboxs[0]

                # Extract region of interest (face)
                x, y, w, h = bbox["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                face_roi = frame[y: y + h, x: x + w]

                # If ROI is valid, get simple features (resize to standard size)
                if face_roi.size > 0:
                    # Resize to standard size for feature consistency
                    face_resized = cv2.resize(face_roi, (100, 100))

                    # Convert to grayscale for simpler features
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

                    # Flatten and normalize as a basic feature vector
                    face_feature = face_gray.flatten() / 255.0

                    # Add a debug message if face is too close
                    if is_too_close:
                        print("FACE TOO CLOSE TO CAMERA - Marking as fake")

                    # Return feature vector, bbox, and real face flag
                    return face_feature, bbox, is_real_face

        except Exception as e:
            print(f"Error in face detection: {e}")
            traceback.print_exc()

        return None, None, False



class ProcessingThread(QThread):
    result_signal = pyqtSignal(object, object, bool)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.frame = None
        self.running = True

    def set_frame(self, frame):
        self.frame = frame.copy() if frame is not None else None

    def run(self):
        global detector

        try:
            # Initialize face detector in this thread
            if detector is None:
                detector = initialize_face_detector()
                if detector is None:
                    self.error_signal.emit(
                        "Could not initialize face detector"
                    )
                    return

            while self.running:
                if self.frame is not None:
                    frame = self.frame.copy()
                    (
                        face_feature,
                        face_bbox,
                        is_real_face,
                    ) = FaceDetection.detect_face_and_antispoofing(frame)
                    self.result_signal.emit(
                        face_feature, face_bbox, is_real_face
                    )

                # Sleep to avoid high CPU usage
                self.msleep(200)  # Process every 200ms

        except Exception as e:
            self.error_signal.emit(f"Processing thread error: {str(e)}")
            traceback.print_exc()

    def stop(self):
        self.running = False
        self.wait()


class AttendanceReportDialog(QDialog):
    def __init__(self, attendance_data, parent=None):
        super().__init__(parent)
        self.attendance_data = attendance_data
        self.setWindowTitle("Today's Attendance Report")
        self.setFixedSize(500, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Attendance Report")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #2C3E50; margin-bottom: 10px;"
        )
        layout.addWidget(title)

        # Date
        date_label = QLabel(f"Date: {time.strftime('%Y-%m-%d')}")
        date_label.setAlignment(Qt.AlignCenter)
        date_label.setStyleSheet("font-size: 16px; color: #34495E;")
        layout.addWidget(date_label)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #BDC3C7; margin: 10px 0px;")
        layout.addWidget(line)

        # Attendance list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "border: 1px solid #E0E0E0; border-radius: 5px;"
        )

        content_widget = QWidget()
        scroll_layout = QVBoxLayout(content_widget)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header_layout = QHBoxLayout()
        name_header = QLabel("Name")
        name_header.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #2980B9;"
        )
        time_header = QLabel("Time")
        time_header.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #2980B9;"
        )

        header_layout.addWidget(name_header)
        header_layout.addWidget(time_header, alignment=Qt.AlignRight)
        scroll_layout.addLayout(header_layout)

        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #BDC3C7;")
        scroll_layout.addWidget(line)

        # Add attendance entries
        if self.attendance_data:
            # Sort by time
            sorted_attendance = sorted(
                self.attendance_data.items(), key=lambda x: x[1]
            )

            for i, (name, time_str) in enumerate(sorted_attendance):
                entry_layout = QHBoxLayout()

                # Alternate row colors
                bg_color = "#F5F8FA" if i % 2 == 0 else "#FFFFFF"

                row_widget = QWidget()
                row_widget.setStyleSheet(
                    f"background-color: {bg_color}; border-radius: 4px;"
                )
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(10, 8, 10, 8)

                name_label = QLabel(name)
                name_label.setStyleSheet("font-size: 15px; color: #2C3E50;")
                time_label = QLabel(time_str)
                time_label.setStyleSheet("font-size: 15px; color: #2C3E50;")

                row_layout.addWidget(name_label)
                row_layout.addWidget(time_label, alignment=Qt.AlignRight)

                scroll_layout.addWidget(row_widget)
        else:
            no_data = QLabel("No attendance records for today")
            no_data.setAlignment(Qt.AlignCenter)
            no_data.setStyleSheet(
                "font-size: 16px; color: #7F8C8D; padding: 20px;"
            )
            scroll_layout.addWidget(no_data)

        scroll_layout.addStretch()
        scroll_area.setWidget(content_widget)

        layout.addWidget(scroll_area)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1F618D;
            }
        """
        )
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.attendance_system = AttendanceSystem()

        # Set application style
        self.setup_style()

        # Initialize camera variables
        self.available_cameras = self.get_available_cameras()
        self.current_camera_id = 0 if self.available_cameras else -1

        # Initialize UI
        self.init_ui()

        # Initialize video and processing threads
        self.video_thread = None
        self.processing_thread = None

        # Current frame and mode
        self.current_frame = None
        self.mode = None  # 'register' or 'login'

        # Face detection results
        self.current_face_feature = None
        self.current_face_bbox = None
        self.is_real_face = False
        self.recognized_name = None

        # Recognition cooldown to prevent multiple recognitions in quick succession
        self.last_recognition_time = 0
        self.recognition_cooldown = 3  # seconds

        # Anti-spoofing verification timer variables
        self.real_face_start_time = 0  # When face first detected as real
        self.fake_face_start_time = 0  # When face first detected as fake
        self.real_face_duration = 3.0  # Required duration in seconds for real face
        self.fake_face_duration = 3.0  # Required duration in seconds for fake face
        self.real_face_verified = False  # Whether real face has been verified
        self.fake_face_verified = False  # Whether fake face has been verified
        self.last_status = None

        # Initialize processing thread
        self.initialize_processing_thread()

    def update_face_results(self, face_feature, face_bbox, is_real_face):
        """Update face detection and anti-spoofing results"""
        # Original functionality
        self.current_face_feature = face_feature
        self.current_face_bbox = face_bbox

        # Add new timing functionality for continuous verification
        current_time = time.time()

        # No face detected case
        if face_bbox is None:
            self.last_status = None
            self.real_face_start_time = 0
            self.fake_face_start_time = 0
            self.real_face_verified = False
            self.fake_face_verified = False
            self.is_real_face = False
            self.recognized_name = None  # Reset recognized name when no face is detected
            if hasattr(self, '_fake_alert_shown'):
                delattr(self, '_fake_alert_shown')  # Reset fake alert shown flag
            return

        # Check if the face status has changed
        if is_real_face != self.last_status:
            # Status changed, reset appropriate timer
            if is_real_face:
                # Switched to real face
                self.real_face_start_time = current_time
                self.real_face_verified = False
                self.fake_face_start_time = 0
                self.fake_face_verified = False
                if hasattr(self, '_fake_alert_shown'):
                    delattr(self, '_fake_alert_shown')  # Reset fake alert shown flag
            else:
                # Switched to fake face
                self.fake_face_start_time = current_time
                self.fake_face_verified = False
                self.real_face_start_time = 0
                self.real_face_verified = False

            self.last_status = is_real_face

        # For real face detection
        if is_real_face and not self.real_face_verified:
            elapsed_time = current_time - self.real_face_start_time
            # If enough time has passed, mark as verified
            if elapsed_time >= self.real_face_duration:
                self.real_face_verified = True
                print(f"Face verified as real after {elapsed_time:.1f} seconds")

        # For fake face detection
        elif not is_real_face and not self.fake_face_verified:
            elapsed_time = current_time - self.fake_face_start_time
            # If enough time has passed, mark as verified and show alert
            if elapsed_time >= self.fake_face_duration:
                self.fake_face_verified = True
                print(f"Face verified as fake after {elapsed_time:.1f} seconds")
                # Show fake face alert after verification (only once)
                if self.mode == "login" and not hasattr(self, '_fake_alert_shown'):
                    self._fake_alert_shown = True
                    # Use QTimer to show the dialog in the main thread after a slight delay
                    QTimer.singleShot(200, self.show_fake_face_alert)

        # Set the real face flag based on verification
        self.is_real_face = self.real_face_verified


    def get_available_cameras(self):
        """Get a list of available camera indices"""
        available_cameras = []
        # Check the first 5 indices (0-4) for available cameras
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        print(f"Available cameras: {available_cameras}")
        return available_cameras

    def setup_style(self):
        # Set application style
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #F5F5F5;
            }
            QLabel {
                color: #2C3E50;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 15px;
                font-weight: bold;
                border-radius: 5px;
                min-width: 120px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1F618D;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                background-color: white;
                font-size: 15px;
            }
            QLineEdit:focus {
                border: 1px solid #3498DB;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                background-color: white;
                font-size: 15px;
                min-width: 180px;
            }
            QComboBox:focus {
                border: 1px solid #3498DB;
            }
        """
        )

    def initialize_processing_thread(self):
        # Initialize processing thread
        self.processing_thread = ProcessingThread()
        self.processing_thread.result_signal.connect(self.update_face_results)
        self.processing_thread.error_signal.connect(self.handle_error)
        self.processing_thread.start()
        print("Processing thread started")

    def initialize_video_thread(self):
        # Stop existing video thread if running
        if self.video_thread and self.video_thread.isRunning():
            print("Stopping existing video thread before initializing a new one")
            self.video_thread.stop()
            # Wait briefly to ensure thread has stopped
            self.video_thread.wait(500)  # Wait for 500ms
            self.video_thread = None

        # Clear any cached frame display before starting a new camera
        if self.mode == "login":
            self.login_camera_label.clear()
            self.login_camera_label.setText("Connecting to camera...")
        elif self.mode == "register":
            self.register_camera_label.clear()
            self.register_camera_label.setText("Connecting to camera...")

        # Reset detection state
        self.current_face_feature = None
        self.current_face_bbox = None
        self.is_real_face = False
        self.current_frame = None

        # Initialize new video thread with current camera
        if self.current_camera_id >= 0:
            print(f"Initializing new video thread with camera {self.current_camera_id}")
            self.video_thread = VideoThread(self.current_camera_id)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.error_signal.connect(self.handle_error)
            self.video_thread.camera_status_signal.connect(
                self.handle_camera_status
            )
            self.video_thread.start()
            return True
        else:
            self.handle_error("No camera available")
            return False

    def handle_camera_status(self, success, message):
        """Handle camera status updates"""
        if self.mode == "login":
            if success:
                self.login_status_label.setText(
                    "Camera connected. Please look at the camera."
                )
            else:
                self.login_status_label.setText(f"Camera issue: {message}")
        elif self.mode == "register":
            if success:
                self.register_status_label.setText(
                    "Camera connected. Please look at the camera."
                )
            else:
                self.register_status_label.setText(f"Camera issue: {message}")

    def init_ui(self):
        self.setWindowTitle("Classroom Attendance System")
        self.setGeometry(100, 100, 900, 700)

        # Create stacked widget for multiple pages
        self.stacked_widget = QStackedWidget()

        # Create pages
        self.create_welcome_page()
        self.create_login_page()
        self.create_register_page()

        self.setCentralWidget(self.stacked_widget)

    def create_welcome_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Add title
        title = QLabel("Classroom Attendance System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #2C3E50; margin: 30px;"
        )
        layout.addWidget(title)

        # Add subtitle or description
        subtitle = QLabel("Secure face recognition attendance tracking")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "font-size: 18px; color: #7F8C8D; margin-bottom: 40px;"
        )
        layout.addWidget(subtitle)

        # Add camera selection
        camera_frame = QFrame()
        camera_frame.setStyleSheet(
            "background-color: #ECF0F1; border-radius: 8px; padding: 10px;"
        )
        camera_layout = QHBoxLayout(camera_frame)

        camera_label = QLabel("Select Camera:")
        camera_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.camera_combo = QComboBox()
        if self.available_cameras:
            for cam_id in self.available_cameras:
                self.camera_combo.addItem(f"Camera {cam_id}", cam_id)
        else:
            self.camera_combo.addItem("No cameras found", -1)

        self.camera_combo.currentIndexChanged.connect(self.camera_changed)

        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)

        layout.addWidget(camera_frame)

        # Add buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(15)

        login_btn = QPushButton("Quick Login")
        login_btn.setFixedSize(250, 60)
        login_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3498DB;
                color: white;
                font-size: 18px;
                border-radius: 8px;
            }
        """
        )
        login_btn.clicked.connect(self.show_login_page)

        register_btn = QPushButton("Register New User")
        register_btn.setFixedSize(250, 60)
        register_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2ECC71;
                color: white;
                font-size: 18px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """
        )
        register_btn.clicked.connect(self.show_register_page)

        view_report_btn = QPushButton("View Today's Attendance")
        view_report_btn.setFixedSize(250, 60)
        view_report_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #9B59B6;
                color: white;
                font-size: 18px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """
        )
        view_report_btn.clicked.connect(self.show_attendance_report)

        btn_layout.addWidget(login_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(register_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(view_report_btn, alignment=Qt.AlignCenter)

        layout.addLayout(btn_layout)

        # Status
        registered_users = self.attendance_system.get_registered_users()
        registered_count = len(registered_users)

        status_frame = QFrame()
        status_frame.setStyleSheet(
            "background-color: #ECF0F1; border-radius: 8px; padding: 15px; margin-top: 20px;"
        )
        status_layout = QVBoxLayout(status_frame)

        registered_label = QLabel(f"Registered Users: {registered_count}")
        registered_label.setStyleSheet("font-size: 16px; color: #2C3E50;")

        today_count = len(self.attendance_system.get_todays_attendance())
        today_label = QLabel(f"Today's Attendance: {today_count}")
        today_label.setStyleSheet("font-size: 16px; color: #2C3E50;")

        status_layout.addWidget(registered_label)
        status_layout.addWidget(today_label)

        layout.addWidget(status_frame)
        layout.addStretch()

        # Version info
        version_label = QLabel("v1.0.0")
        version_label.setAlignment(Qt.AlignRight)
        version_label.setStyleSheet("color: #95A5A6; font-size: 12px;")
        layout.addWidget(version_label)

        page.setLayout(layout)
        self.stacked_widget.addWidget(page)
        self.welcome_page = page

    def camera_changed(self, index):
        """Handle camera selection change"""
        self.current_camera_id = self.camera_combo.currentData()
        print(f"Camera changed to {self.current_camera_id}")

        # Restart video thread if active
        if self.video_thread and self.video_thread.isRunning():
            if self.mode in ['login', 'register']:
                self.initialize_video_thread()

    def create_login_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Title
        title = QLabel("Quick Face Login")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2C3E50; margin-bottom: 20px;")
        layout.addWidget(title)

        # Camera view
        self.login_camera_label = QLabel()
        self.login_camera_label.setFixedSize(640, 480)
        self.login_camera_label.setAlignment(Qt.AlignCenter)
        self.login_camera_label.setStyleSheet("border: 2px solid #BDC3C7; background-color: #EAEAEA;")
        self.login_camera_label.setText("Camera view will appear here")
        layout.addWidget(self.login_camera_label, alignment=Qt.AlignCenter)

        # Status and detection info
        self.login_status_label = QLabel("Look at the camera for face detection")
        self.login_status_label.setAlignment(Qt.AlignCenter)
        self.login_status_label.setStyleSheet("font-size: 16px; color: #7F8C8D; margin: 10px 0;")
        layout.addWidget(self.login_status_label)

        # Recognition status
        self.login_recognition_frame = QFrame()
        self.login_recognition_frame.setVisible(False)
        self.login_recognition_frame.setStyleSheet(
            "background-color: #D5F5E3; border-radius: 8px; padding: 15px; margin: 10px 0;"
        )
        recognition_layout = QVBoxLayout(self.login_recognition_frame)

        self.login_recognition_name = QLabel()
        self.login_recognition_name.setStyleSheet("font-size: 18px; font-weight: bold; color: #27AE60;")
        self.login_recognition_confidence = QLabel()
        self.login_recognition_confidence.setStyleSheet("font-size: 16px; color: #2C3E50;")
        self.login_recognition_time = QLabel()
        self.login_recognition_time.setStyleSheet("font-size: 16px; color: #2C3E50;")

        recognition_layout.addWidget(self.login_recognition_name)
        recognition_layout.addWidget(self.login_recognition_confidence)
        recognition_layout.addWidget(self.login_recognition_time)

        layout.addWidget(self.login_recognition_frame)

        # Buttons
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #95A5A6;
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
            """
        )
        back_btn.clicked.connect(self.show_welcome_page)


        btn_layout.addWidget(back_btn)

        layout.addLayout(btn_layout)

        page.setLayout(layout)
        self.stacked_widget.addWidget(page)
        self.login_page = page

    def create_register_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Title
        title = QLabel("Register New User")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2C3E50; margin-bottom: 20px;")
        layout.addWidget(title)

        # Form and camera side by side
        form_camera_layout = QHBoxLayout()

        # Form
        form_frame = QFrame()
        form_frame.setStyleSheet("background-color: #ECF0F1; border-radius: 8px; padding: 20px;")
        form_layout = QVBoxLayout(form_frame)

        username_label = QLabel("Username:")
        username_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("Enter username")

        password_label = QLabel("Password:")
        password_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.register_password = QLineEdit()
        self.register_password.setEchoMode(QLineEdit.Password)
        self.register_password.setPlaceholderText("Enter password")

        confirm_label = QLabel("Confirm Password:")
        confirm_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.register_confirm = QLineEdit()
        self.register_confirm.setEchoMode(QLineEdit.Password)
        self.register_confirm.setPlaceholderText("Confirm password")

        register_btn = QPushButton("Register")
        register_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2ECC71;
                color: white;
                font-size: 16px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            """
        )
        register_btn.clicked.connect(self.register_user)

        form_layout.addWidget(username_label)
        form_layout.addWidget(self.register_username)
        form_layout.addWidget(password_label)
        form_layout.addWidget(self.register_password)
        form_layout.addWidget(confirm_label)
        form_layout.addWidget(self.register_confirm)
        form_layout.addStretch()
        form_layout.addWidget(register_btn)

        # Camera view
        camera_frame = QFrame()
        camera_layout = QVBoxLayout(camera_frame)

        camera_label = QLabel("Face Registration:")
        camera_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.register_camera_label = QLabel()
        self.register_camera_label.setFixedSize(320, 240)
        self.register_camera_label.setAlignment(Qt.AlignCenter)
        self.register_camera_label.setStyleSheet("border: 2px solid #BDC3C7; background-color: #EAEAEA;")
        self.register_camera_label.setText("Camera view will appear here")

        self.register_status_label = QLabel("Look at the camera for face detection")
        self.register_status_label.setAlignment(Qt.AlignCenter)
        self.register_status_label.setStyleSheet("font-size: 14px; color: #7F8C8D; margin: 10px 0;")

        face_status_frame = QFrame()
        face_status_frame.setStyleSheet("background-color: #ECF0F1; border-radius: 5px; padding: 10px;")
        face_status_layout = QHBoxLayout(face_status_frame)

        self.register_face_status = QLabel("No face detected")
        self.register_face_status.setStyleSheet("color: #E74C3C;")

        self.register_real_status = QLabel("Anti-spoofing check: Waiting")
        self.register_real_status.setStyleSheet("color: #7F8C8D;")

        face_status_layout.addWidget(self.register_face_status)
        face_status_layout.addWidget(self.register_real_status)

        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.register_camera_label, alignment=Qt.AlignCenter)
        camera_layout.addWidget(self.register_status_label)
        camera_layout.addWidget(face_status_frame)

        form_camera_layout.addWidget(form_frame)
        form_camera_layout.addWidget(camera_frame)

        layout.addLayout(form_camera_layout)

        # Buttons
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #95A5A6;
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
            """
        )
        back_btn.clicked.connect(self.show_welcome_page)

        layout.addWidget(back_btn, alignment=Qt.AlignLeft)

        page.setLayout(layout)
        self.stacked_widget.addWidget(page)
        self.register_page = page



    def update_image(self, frame):
        """Update the camera image in UI"""
        if frame is None:
            return

        self.current_frame = frame.copy()

        # Pass frame to processing thread
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.set_frame(frame)

        # Draw detection box if face detected
        display_frame = frame.copy()
        if self.current_face_bbox is not None:
            x, y, w, h = self.current_face_bbox["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Set color and text based on face verification status
            current_time = time.time()

            # Default status (no verification running)
            color = (128, 128, 128)  # Gray
            status_text = "Detecting..."

            # Check if face is too close
            frame_height, frame_width = frame.shape[:2]
            width_ratio = w / frame_width
            height_ratio = h / frame_height
            is_too_close = (width_ratio > 0.6) or (height_ratio > 0.6)

            # Real face verification in progress
            if self.last_status is True and not self.real_face_verified:
                elapsed_time = current_time - self.real_face_start_time
                remaining = max(0, self.real_face_duration - elapsed_time)
                color = (255, 165, 0)  # Orange for verification in progress
                status_text = f"Verifying Real: {remaining:.1f}s"

                # Add too close warning
                if is_too_close:
                    status_text = "TOO CLOSE TO CAMERA!"
                    color = (0, 0, 255)  # Red for too close warning

            # Fake face verification in progress
            elif self.last_status is False and not self.fake_face_verified:
                elapsed_time = current_time - self.fake_face_start_time
                remaining = max(0, self.fake_face_duration - elapsed_time)

                if is_too_close:
                    color = (0, 0, 255)  # Red for too close
                    status_text = "TOO CLOSE TO CAMERA!"
                else:
                    color = (255, 0, 255)  # Purple for fake verification
                    status_text = f"Verifying Fake: {remaining:.1f}s"

            # Verification complete
            elif self.real_face_verified:
                if is_too_close:
                    # Override verified status if too close now
                    color = (0, 0, 255)  # Red for too close
                    status_text = "TOO CLOSE TO CAMERA!"
                else:
                    color = (0, 255, 0)  # Green for verified real face
                    status_text = "Real Face Verified"
            elif self.fake_face_verified:
                color = (0, 0, 255)  # Red for verified fake face
                status_text = "Fake Face Verified"

                # Add too close message if applicable
                if is_too_close:
                    status_text = "TOO CLOSE TO CAMERA!"

            # Draw rectangle and status text
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, status_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # If recognized, show name
            if self.recognized_name:
                cv2.putText(display_frame, self.recognized_name, (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Convert to QImage and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Resize and display based on mode (login or register)
        if self.mode == "login":
            p = convert_to_qt_format.scaled(self.login_camera_label.width(), self.login_camera_label.height(),
                                            Qt.KeepAspectRatio)
            self.login_camera_label.setPixmap(QPixmap.fromImage(p))

            # Update status text for login page
            self.update_login_status_text()

        elif self.mode == "register":
            p = convert_to_qt_format.scaled(self.register_camera_label.width(), self.register_camera_label.height(),
                                            Qt.KeepAspectRatio)
            self.register_camera_label.setPixmap(QPixmap.fromImage(p))

            # Update status text for register page
            self.update_register_status_text()

        # In login mode, check for face recognition
        if self.mode == "login":
            self.attempt_recognition()


    def update_login_status_text(self):
        """Update status text on the login page based on face verification status"""
        current_time = time.time()

        # No face detected
        if self.current_face_bbox is None:
            self.login_status_label.setText("No face detected. Please look at the camera.")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #7F8C8D;")
            return

        # Check if face is too close to camera
        if self.current_face_bbox is not None:
            x, y, w, h = self.current_face_bbox["bbox"]
            frame_height, frame_width = self.current_frame.shape[:2] if self.current_frame is not None else (480, 640)
            width_ratio = w / frame_width
            height_ratio = h / frame_height
            is_too_close = (width_ratio > 0.6) or (height_ratio > 0.6)

            if is_too_close:
                self.login_status_label.setText("Face is TOO CLOSE to camera! Please move back.")
                self.login_status_label.setStyleSheet("font-size: 16px; color: #E74C3C; font-weight: bold;")
                return

        # Real face verification in progress
        if self.last_status is True and not self.real_face_verified:
            elapsed_time = current_time - self.real_face_start_time
            remaining = max(0, self.real_face_duration - elapsed_time)
            self.login_status_label.setText(f"Verifying real face: {remaining:.1f}s remaining")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #F39C12;")  # Orange

        # Fake face verification in progress
        elif self.last_status is False and not self.fake_face_verified:
            elapsed_time = current_time - self.fake_face_start_time
            remaining = max(0, self.fake_face_duration - elapsed_time)
            self.login_status_label.setText(f"Detecting fake face: {remaining:.1f}s remaining")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #9B59B6;")  # Purple

        # Real face verified
        elif self.real_face_verified:
            self.login_status_label.setText("Face verified as real. Recognizing...")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #27AE60;")  # Green

        # Fake face verified
        elif self.fake_face_verified:
            self.login_status_label.setText("Fake face detected. Please use a real face.")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #E74C3C;")  # Red

        # Face detected but verification not started yet
        else:
            self.login_status_label.setText("Face detected. Starting verification...")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #7F8C8D;")  # Gray

    def update_register_status_text(self):
        """Update status text on the register page based on face verification status"""
        current_time = time.time()

        # No face detected
        if self.current_face_bbox is None:
            self.register_face_status.setText("No face detected")
            self.register_face_status.setStyleSheet("color: #E74C3C;")  # Red
            self.register_real_status.setText("Anti-spoofing: Waiting")
            self.register_real_status.setStyleSheet("color: #7F8C8D;")  # Gray
            return

        # Check if face is too close to camera
        if self.current_face_bbox is not None:
            x, y, w, h = self.current_face_bbox["bbox"]
            frame_height, frame_width = self.current_frame.shape[:2] if self.current_frame is not None else (480, 640)
            width_ratio = w / frame_width
            height_ratio = h / frame_height
            is_too_close = (width_ratio > 0.6) or (height_ratio > 0.6)

            if is_too_close:
                self.register_face_status.setText("Face detected (TOO CLOSE!)")
                self.register_face_status.setStyleSheet("color: #E74C3C; font-weight: bold;")  # Red
                self.register_real_status.setText("Move back from camera!")
                self.register_real_status.setStyleSheet("color: #E74C3C; font-weight: bold;")  # Red
                self.register_status_label.setText("Face too close to camera. Please move back.")
                self.register_status_label.setStyleSheet("font-size: 14px; color: #E74C3C; font-weight: bold;")
                return

        # Face detected - update status
        self.register_face_status.setText("Face detected")
        self.register_face_status.setStyleSheet("color: #27AE60;")  # Green

        # Real face verification in progress
        if self.last_status is True and not self.real_face_verified:
            elapsed_time = current_time - self.real_face_start_time
            remaining = max(0, self.real_face_duration - elapsed_time)
            self.register_real_status.setText(f"Verifying real: {remaining:.1f}s")
            self.register_real_status.setStyleSheet("color: #F39C12;")  # Orange
            self.register_status_label.setText("Keep facing the camera...")
            self.register_status_label.setStyleSheet("font-size: 14px; color: #F39C12;")

        # Fake face verification in progress
        elif self.last_status is False and not self.fake_face_verified:
            elapsed_time = current_time - self.fake_face_start_time
            remaining = max(0, self.fake_face_duration - elapsed_time)
            self.register_real_status.setText(f"Verifying fake: {remaining:.1f}s")
            self.register_real_status.setStyleSheet("color: #9B59B6;")  # Purple
            self.register_status_label.setText("Potential fake face detected...")
            self.register_status_label.setStyleSheet("font-size: 14px; color: #9B59B6;")

        # Real face verified
        elif self.real_face_verified:
            self.register_real_status.setText("Anti-spoofing: Passed")
            self.register_real_status.setStyleSheet("color: #27AE60;")  # Green
            self.register_status_label.setText("Face verified! You can now complete registration.")
            self.register_status_label.setStyleSheet("font-size: 14px; color: #27AE60; font-weight: bold;")

        # Fake face verified
        elif self.fake_face_verified:
            self.register_real_status.setText("Anti-spoofing: Failed")
            self.register_real_status.setStyleSheet("color: #E74C3C;")  # Red
            self.register_status_label.setText("Fake face detected. Please use a real face.")
            self.register_status_label.setStyleSheet("font-size: 14px; color: #E74C3C;")

        # Face detected but verification not started yet
        else:
            self.register_real_status.setText("Anti-spoofing: Starting...")
            self.register_real_status.setStyleSheet("color: #7F8C8D;")  # Gray
            self.register_status_label.setText("Please keep looking at the camera")
            self.register_status_label.setStyleSheet("font-size: 14px; color: #7F8C8D;")

    def attempt_recognition(self):
        """Attempt to recognize the face in login mode"""
        # Only attempt recognition if we have a verified real face and enough time elapsed since last recognition
        current_time = time.time()

        if (self.mode == "login" and self.current_face_feature is not None and
                self.is_real_face):

            # Check if face is too close to camera
            if self.current_face_bbox is not None:
                x, y, w, h = self.current_face_bbox["bbox"]
                frame_height, frame_width = self.current_frame.shape[:2] if self.current_frame is not None else (
                480, 640)
                width_ratio = w / frame_width
                height_ratio = h / frame_height
                is_too_close = (width_ratio > 0.6) or (height_ratio > 0.6)

                if is_too_close:
                    # Don't attempt recognition if face is too close
                    self.login_status_label.setText("Face is TOO CLOSE to camera! Please move back.")
                    self.login_status_label.setStyleSheet("font-size: 16px; color: #E74C3C; font-weight: bold;")
                    return

            # Only attempt a new recognition if enough cooldown time has passed
            if (current_time - self.last_recognition_time) > self.recognition_cooldown:
                success, username, confidence, timestamp = self.attendance_system.authenticate_user(
                    self.current_face_feature)

                if success:
                    self.recognized_name = username
                    self.last_recognition_time = current_time

                    # Update login UI
                    self.login_recognition_frame.setVisible(True)
                    self.login_recognition_name.setText(f"Welcome, {username}!")
                    self.login_recognition_confidence.setText(f"Confidence: {confidence:.1f}%")
                    self.login_recognition_time.setText(f"Attendance recorded at: {timestamp}")

                    # Update status
                    self.login_status_label.setText(f"Successfully recognized {username}. Attendance recorded.")
                    self.login_status_label.setStyleSheet("font-size: 16px; color: #27AE60; font-weight: bold;")

                    # Show success message with OK button instead of auto-navigating to dashboard
                    self.show_attendance_success_dialog(username, timestamp)
                else:
                    # Clear previous recognition when a new face is not recognized
                    self.recognized_name = None
                    if username == "Face not recognized":
                        self.login_status_label.setText("Face not recognized. Please try again.")
                        self.login_status_label.setStyleSheet("font-size: 16px; color: #E74C3C;")

    def show_attendance_success_dialog(self, username, timestamp):
        """Show a success dialog when attendance is marked successfully"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Attendance Marked")
        msg.setText(f"Attendance marked successfully for {username}")
        msg.setInformativeText(f"Time: {timestamp}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QLabel {
                color: #27AE60;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)

        # When user clicks OK, reset the recognition state for next person
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.reset_login_state()

    def reset_registered_students(self):
        """Reset all registered student data"""
        # Create a confirmation dialog
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Reset All Users")
        msg.setText("Are you sure you want to reset all registered users?")
        msg.setInformativeText("This will delete all face data and user credentials. This action cannot be undone!")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)  # Default is No to prevent accidental deletion
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QLabel {
                color: #E74C3C;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)

        # Execute the dialog and get user response
        result = msg.exec_()

        # If the user clicks Yes, reset the data
        if result == QMessageBox.Yes:
            try:
                # Clear user data in memory
                self.attendance_system.known_face_features = []
                self.attendance_system.known_face_names = []
                self.attendance_system.user_credentials = {}

                # Save the empty data to persist the reset
                self.attendance_system.save_data()

                # Show success message
                success_msg = QMessageBox(self)
                success_msg.setIcon(QMessageBox.Information)
                success_msg.setWindowTitle("Reset Complete")
                success_msg.setText("All registered users have been removed successfully.")
                success_msg.setStandardButtons(QMessageBox.Ok)
                success_msg.setStyleSheet("""
                    QMessageBox {
                        background-color: #FFFFFF;
                    }
                    QLabel {
                        color: #27AE60;
                        font-size: 16px;
                    }
                    QPushButton {
                        background-color: #3498DB;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        font-size: 14px;
                        border-radius: 4px;
                        min-width: 80px;
                    }
                    QPushButton:hover {
                        background-color: #2980B9;
                    }
                """)
                success_msg.exec_()

                return True
            except Exception as e:
                # Show error message if deletion fails
                error_msg = QMessageBox(self)
                error_msg.setIcon(QMessageBox.Critical)
                error_msg.setWindowTitle("Error")
                error_msg.setText("Failed to reset registered users")
                error_msg.setInformativeText(f"Error: {str(e)}")
                error_msg.setStandardButtons(QMessageBox.Ok)
                error_msg.exec_()
                return False

        return False


    def show_fake_face_alert(self):
        """Show alert when a fake face is detected"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Fake Face Detected")
        msg.setText("Anti-spoofing check failed")
        msg.setInformativeText("Please use a real face for authentication")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QLabel {
                color: #E74C3C;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)

        # When user clicks OK, reset the recognition state for next person
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.reset_login_state()

    def reset_login_state(self):
        """Reset login state to prepare for the next person"""
        # Reset recognition and verification state
        self.current_face_feature = None
        self.current_face_bbox = None
        self.is_real_face = False
        self.real_face_verified = False
        self.fake_face_verified = False
        self.last_status = None
        self.recognized_name = None
        self.real_face_start_time = 0
        self.fake_face_start_time = 0

        # Reset fake alert flag
        if hasattr(self, '_fake_alert_shown'):
            delattr(self, '_fake_alert_shown')

        # Reset UI elements
        self.login_recognition_frame.setVisible(False)
        self.login_status_label.setText("Ready for next person. Look at the camera.")
        self.login_status_label.setStyleSheet("font-size: 16px; color: #7F8C8D;")

    def register_user(self):
        """Register a new user"""
        username = self.register_username.text().strip()
        password = self.register_password.text()
        confirm = self.register_confirm.text()

        # Validate input
        if not username:
            self.show_error("Username cannot be empty")
            return

        if not password:
            self.show_error("Password cannot be empty")
            return

        if password != confirm:
            self.show_error("Passwords do not match")
            return

        # Check face detection
        if self.current_face_feature is None:
            self.show_error("No face detected. Please look at the camera")
            return

        # Check if face is too close to camera
        if self.current_face_bbox is not None:
            x, y, w, h = self.current_face_bbox["bbox"]
            frame_height, frame_width = self.current_frame.shape[:2] if self.current_frame is not None else (480, 640)
            width_ratio = w / frame_width
            height_ratio = h / frame_height
            is_too_close = (width_ratio > 0.6) or (height_ratio > 0.6)

            if is_too_close:
                self.show_error("Face is too close to the camera. Please move back and try again.")
                return

        # Check real face - uses the verified real face status
        if not self.is_real_face:
            self.show_error("Anti-spoofing check failed. Face must be verified as real for 3 seconds")
            return

        # Register user
        success, message = self.attendance_system.register_user(username, password, self.current_face_feature)

        if success:
            self.show_message("Success", "User registered successfully")
            self.register_username.clear()
            self.register_password.clear()
            self.register_confirm.clear()
            self.show_welcome_page()
        else:
            self.show_error(message)
    def update_time(self):
        """Update the time display on dashboard"""
        current_time = time.strftime("%H:%M:%S")


    def show_welcome_page(self):
        """Show welcome page and stop camera threads"""
        # Reset the fake alert flag when returning to welcome page
        if hasattr(self, '_fake_alert_shown'):
            delattr(self, '_fake_alert_shown')

        # Continue with original method...
        # Stop video thread
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None

        # Reset state
        self.mode = None
        self.current_frame = None
        self.current_face_feature = None
        self.current_face_bbox = None
        self.is_real_face = False
        self.recognized_name = None

        # Show page
        self.stacked_widget.setCurrentWidget(self.welcome_page)

    def show_login_page(self):
        """Show login page and start camera"""
        # Reset the fake alert flag when going to login page
        if hasattr(self, '_fake_alert_shown'):
            delattr(self, '_fake_alert_shown')

        self.mode = "login"

        # Reset UI elements
        self.login_camera_label.setText("Connecting to camera...")
        self.login_status_label.setText("Look at the camera for face detection")
        self.login_status_label.setStyleSheet("font-size: 16px; color: #7F8C8D;")
        self.login_recognition_frame.setVisible(False)

        # Start camera
        self.initialize_video_thread()

        # Show page
        self.stacked_widget.setCurrentWidget(self.login_page)

    def show_register_page(self):
        """Show register page and start camera"""
        self.mode = "register"

        # Reset UI elements
        self.register_username.clear()
        self.register_password.clear()
        self.register_confirm.clear()
        self.register_camera_label.setText("Connecting to camera...")
        self.register_status_label.setText("Look at the camera for face detection")
        self.register_face_status.setText("No face detected")
        self.register_face_status.setStyleSheet("color: #E74C3C;")
        self.register_real_status.setText("Anti-spoofing check: Waiting")
        self.register_real_status.setStyleSheet("color: #7F8C8D;")

        # Start camera
        self.initialize_video_thread()

        # Show page
        self.stacked_widget.setCurrentWidget(self.register_page)


    def show_attendance_report(self):
        """Show attendance report dialog"""
        today_attendance = self.attendance_system.get_todays_attendance()
        dialog = AttendanceReportDialog(today_attendance, self)
        dialog.exec_()

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)

    def show_message(self, title, message):
        """Show information message"""
        QMessageBox.information(self, title, message)

    def handle_error(self, error_message):
        """Handle error messages from threads"""
        print(f"Error: {error_message}")
        if self.mode == "login":
            self.login_status_label.setText(f"Error: {error_message}")
            self.login_status_label.setStyleSheet("font-size: 16px; color: #E74C3C;")
        elif self.mode == "register":
            self.register_status_label.setText(f"Error: {error_message}")
            self.register_status_label.setStyleSheet("font-size: 14px; color: #E74C3C;")

    def closeEvent(self, event):
        """Handle application close event"""
        # Stop threads
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()

        # Accept the close event
        event.accept()

if __name__ == "__main__":
        # Create application
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle("Fusion")

        # Create and show main window
        main_window = MainWindow()
        main_window.show()

        # Run application
        sys.exit(app.exec_())
