import sys
import cv2
import numpy as np
import face_recognition
import os
import pickle
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QProgressDialog,
    QLabel, QVBoxLayout, QHBoxLayout, QInputDialog, QMessageBox,
    QTabWidget, QTextEdit, QScrollArea, QGridLayout, QCheckBox,
    QDialog, QDialogButtonBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot
from sklearn.cluster import DBSCAN
from collections import defaultdict
from performance import PerformanceGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceSelectionDialog(QDialog):
    def __init__(self, face_images, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Faces to Register")
        self.selected_faces = []
        self.face_images = face_images[:25]  # Limit to 25 faces

        layout = QVBoxLayout()
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_checkboxes)
        layout.addWidget(self.select_all_checkbox)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        grid = QGridLayout(scroll_widget)
        self.checkboxes = []
        self.face_labels = []

        for i, img in enumerate(self.face_images):
            label = QLabel()
            label.setFixedSize(100, 100)
            label.setPixmap(QPixmap.fromImage(img).scaled(100, 100, Qt.KeepAspectRatio))
            checkbox = QCheckBox(f"Face {i+1}")
            self.checkboxes.append(checkbox)
            self.face_labels.append(label)
            grid.addWidget(label, i // 4, (i % 4) * 2)
            grid.addWidget(checkbox, i // 4, (i % 4) * 2 + 1)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def toggle_all_checkboxes(self, state):
        for cb in self.checkboxes:
            cb.setChecked(state == Qt.Checked)

    def get_selected_indices(self):
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

class FaceProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    faces_extracted = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, min_face_size, sample_every_n=10):
        super().__init__()
        self.video_path = video_path
        self.min_face_size = min_face_size
        self.sample_every_n = sample_every_n
        self.is_canceled = False

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("Failed to open video file.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                count += 1
                if count % self.sample_every_n != 0:
                    continue

                if self.is_canceled:
                    break

                # Resize frame to reduce processing time
                frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb, model='hog')
                faces = [face for face in faces 
                        if (face[2] - face[0]) >= self.min_face_size and (face[1] - face[3]) >= self.min_face_size]
                
                # Use small model for faster encoding
                encodings = face_recognition.face_encodings(rgb, faces, model='small')

                for (top, right, bottom, left), encoding in zip(faces, encodings):
                    face_img = rgb[top:bottom, left:right]
                    if face_img.size == 0:
                        continue
                    thumb = cv2.resize(face_img, (100, 100))
                    qimg = QImage(thumb.data, 100, 100, 3 * 100, QImage.Format_RGB888)
                    frames.append((encoding, qimg))

                progress = int(count / total_frames * 100)
                self.progress_updated.emit(progress)
                
                # Process in batches to manage memory
                if len(frames) > 1000:  # Limit memory usage
                    self.process_clusters(frames)
                    frames = []

            cap.release()

            if frames:
                self.process_clusters(frames)
            else:
                self.error_occurred.emit("No faces detected in video.")

        except Exception as e:
            self.error_occurred.emit(f"Error during face processing: {str(e)}")

    def process_clusters(self, frames):
        encodings = np.array([f[0] for f in frames])
        if len(encodings) == 0:
            return

        # Optimize DBSCAN parameters
        clusterer = DBSCAN(metric='euclidean', eps=0.5, min_samples=2)
        labels = clusterer.fit_predict(encodings)
        clusters = defaultdict(list)

        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(frames[i])

        clustered_frames = [cluster for cluster in clusters.values()]
        self.faces_extracted.emit(clustered_frames)

    def cancel(self):
        self.is_canceled = True

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.video_width = 750
        self.video_height = 650
        self.face_width = 100
        self.face_height = 100
        self.detected_faces = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_count = 0
        self.processing_frame = False
        self.min_face_size = 15
        self.confidence_threshold = 0.5
        self.face_cache = defaultdict(list)
        self.is_processing_faces = False
        self.was_playing = False

        self.load_registered_faces()
        self.initUI()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.performance_data = []
        self.face_thread = None

    def load_registered_faces(self):
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        if not os.path.exists("faceDatabase"):
            os.makedirs("faceDatabase")

        for file in os.listdir("faceDatabase"):
            if file.endswith(".pkl"):
                try:
                    with open(os.path.join("faceDatabase", file), 'rb') as f:
                        records = pickle.load(f)
                        for name, encoding in records:
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(name)
                except Exception as e:
                    logging.error(f"Error loading {file}: {str(e)}")

    def initUI(self):
        self.setWindowTitle("Multi-Threaded Face Recognition System")
        self.setGeometry(100, 100, 800, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #3A3A3A;
                border: 1px solid #4A4A4A;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
            QLabel {
                padding: 2px;
            }
            QTabWidget::pane {
                border: 1px solid #444;
            }
            QTabBar::tab {
                background: #3A3A3A;
                padding: 8px;
                border: 1px solid #444;
            }
            QTabBar::tab:selected {
                background: #505050;
            }
        """)

        main_layout = QVBoxLayout()

        title_label = QLabel("Multi-Threaded Face Recognition")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                color: #FFFFFF;
                border-bottom: 2px solid #444;
            }
        """)
        main_layout.addWidget(title_label)

        self.tabs = QTabWidget()

        self.video_tab = QWidget()
        self.init_video_tab()
        self.tabs.addTab(self.video_tab, "Video Processing")

        self.performance_tab = QWidget()
        self.init_performance_tab()
        self.tabs.addTab(self.performance_tab, "Performance Metrics")

        main_layout.addWidget(self.tabs)
        main_layout.setStretch(1, 1)
        self.setLayout(main_layout)

    def init_video_tab(self):
        layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(self.video_width, self.video_height)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        control_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.load_video)
        control_layout.addWidget(self.upload_button)

        self.extract_button = QPushButton("Register Faces from Video", self)
        self.extract_button.clicked.connect(self.extract_and_cluster_faces)
        self.extract_button.setEnabled(False)
        control_layout.addWidget(self.extract_button)

        layout.addLayout(control_layout)
        self.video_tab.setLayout(layout)

    def init_performance_tab(self):
        layout = QVBoxLayout()

        self.performance_graph = PerformanceGraph(self)
        layout.addWidget(self.performance_graph, stretch=3)

        self.performance_data_display = QTextEdit()
        self.performance_data_display.setReadOnly(True)
        self.performance_data_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #444;
                padding: 5px;
                font-family: monospace;
            }
        """)
        layout.addWidget(self.performance_data_display, stretch=1)

        self.performance_tab.setLayout(layout)

    def update_performance_data(self):
        if not self.performance_graph.cpu_data or not self.performance_graph.mem_data:
            return

        cpu = self.performance_graph.cpu_data[-1] if self.performance_graph.cpu_data else 0
        mem = self.performance_graph.mem_data[-1] if self.performance_graph.mem_data else 0
        accuracy = self.performance_graph.accuracy_data[-1] if self.performance_graph.accuracy_data else 0
        fps = self.performance_graph.fps_data[-1] if self.performance_graph.fps_data else 0

        data_str = (f"CPU Usage: {cpu:.1f}%\n"
                    f"Memory Usage: {mem:.1f} MB\n"
                    f"Recognition Accuracy: {accuracy:.1f}%\n"
                    f"Frame Rate: {fps:.1f} FPS\n\n"
                    f"Registered Faces: {len(self.known_face_names)}")
        self.performance_data_display.setPlainText(data_str)

        self.performance_data.append({
            'time': self.performance_graph.time_counter,
            'cpu': cpu,
            'mem': mem,
            'accuracy': accuracy,
            'fps': fps
        })

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.stop_video()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file.")
                self.cap = None
                return

            self.video_path = file_path
            self.performance_graph.reset_counters()
            self.performance_data.clear()
            self.extract_button.setEnabled(True)
            self.timer.start(33)

    def stop_video(self):
        """Stop video playback and release resources without clearing the display."""
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if self.processing_frame or not self.cap or not self.cap.isOpened() or self.is_processing_faces:
            return

        self.processing_frame = True
        try:
            self.performance_graph.increment_frame_count()

            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return

            self.frame_count += 1
            if self.frame_count % 4 != 0:
                self.processing_frame = False
                return

            frame = cv2.resize(frame, (self.video_width, self.video_height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='small')

            total_faces = len(face_encodings)
            correct_matches = 0

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                color = (0, 0, 255)

                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.5)
                if True in matches:
                    match_index = matches.index(True)
                    name = self.known_face_names[match_index]
                    color = (0, 255, 0)
                    correct_matches += 1

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.performance_graph.update_accuracy(total_faces, correct_matches)
            self.update_performance_data()

            q_img = QImage(frame.data, self.video_width, self.video_height,
                           3 * self.video_width, QImage.Format_BGR888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
        finally:
            self.processing_frame = False

    def extract_and_cluster_faces(self):
        if not hasattr(self, 'video_path'):
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        # Pause video playback to avoid concurrent file access
        self.was_playing = self.timer.isActive()
        self.stop_video()
        self.is_processing_faces = True
        self.extract_button.setEnabled(False)

        self.progress_dialog = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.canceled.connect(self.cancel_face_processing)

        self.face_thread = FaceProcessingThread(self.video_path, self.min_face_size)
        self.face_thread.progress_updated.connect(self.progress_dialog.setValue)
        self.face_thread.faces_extracted.connect(self.handle_faces_extracted)
        self.face_thread.error_occurred.connect(self.handle_processing_error)
        self.face_thread.finished.connect(self.face_processing_finished)
        self.face_thread.start()

    @pyqtSlot(int)
    def update_progress(self, value):
        self.progress_dialog.setValue(value)

    @pyqtSlot(list)
    def handle_faces_extracted(self, clusters):
        self.progress_dialog.close()
        if not clusters:
            QMessageBox.information(self, "No Clusters", "No recognizable face groups found to register.")
            return

        for cluster_id, samples in enumerate(clusters):
            selected = samples[:25] if len(samples) > 25 else samples
            imgs = [s[1] for s in selected]
            dialog = FaceSelectionDialog(imgs, self)
            if dialog.exec_():
                indices = dialog.get_selected_indices()
                if indices:
                    name, ok = QInputDialog.getText(self, "Register Person", f"Enter name for Cluster {cluster_id + 1}:")
                    if ok and name:
                        self.save_to_database(name, [selected[i][0] for i in indices])
        self.load_registered_faces()

    @pyqtSlot(str)
    def handle_processing_error(self, error_msg):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Error", error_msg)

    def face_processing_finished(self):
        self.face_thread = None
        self.is_processing_faces = False
        self.extract_button.setEnabled(True)

        # Resume video playback if it was playing before
        if self.was_playing and hasattr(self, 'video_path'):
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.timer.start(33)

    def cancel_face_processing(self):
        if self.face_thread:
            self.face_thread.cancel()
            self.face_thread.wait()  # Ensure thread is fully terminated
        self.is_processing_faces = False
        self.extract_button.setEnabled(True)
        # Resume video if it was playing
        if self.was_playing and hasattr(self, 'video_path'):
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.timer.start(33)

    def save_to_database(self, name, encodings):
        db_path = os.path.join("faceDatabase", f"{name}.pkl")
        data = []

        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    existing_data = pickle.load(f)
                    data.extend(existing_data)
            except Exception as e:
                logging.error(f"Error loading existing database {db_path}: {str(e)}")

        data.extend([(name, enc) for enc in encodings])

        if len(data) > 25:
            all_encodings = [enc for _, enc in data]
            avg_distances = []
            for enc in all_encodings:
                distances = face_recognition.face_distance(all_encodings, enc)
                avg_distances.append(np.mean(distances))

            best_indices = np.argsort(avg_distances)[:25]
            data = [data[i] for i in best_indices]

        try:
            with open(db_path, 'wb') as f:
                pickle.dump(data, f)
            QMessageBox.information(self, "Saved", f"Saved {len(data)} encodings for '{name}'.")
        except Exception as e:
            logging.error(f"Error saving to database {db_path}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save encodings: {str(e)}")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.stop_video()
            if self.face_thread:
                self.face_thread.cancel()
                self.face_thread.wait()
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())