import sys
import cv2
import numpy as np
import face_recognition
import os
import pickle
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog,QProgressDialog,
                             QLabel, QVBoxLayout, QHBoxLayout, QInputDialog,
                             QMessageBox, QTabWidget, QTextEdit, QScrollArea,
                             QGridLayout, QCheckBox, QDialog, QDialogButtonBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from sklearn.cluster import DBSCAN
from performance import PerformanceGraph
from collections import defaultdict


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
            label.setPixmap(QPixmap.fromImage(img).scaled(100, 100))
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
        self.min_face_size = 15  # Minimum face size in pixels
        self.confidence_threshold = 0.5  # Confidence threshold for face recognition
        self.face_cache = defaultdict(list)  # Cache for storing face encodings per person
    
        self.load_registered_faces()
        self.initUI()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.performance_data = []

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
        self.setWindowTitle("Face Recognition System")
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

        title_label = QLabel("Single-Threaded Face Recognition")
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
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file.")
                return

            self.video_path = file_path
            self.performance_graph.reset_counters()
            self.performance_data.clear()
            self.extract_button.setEnabled(True)
            self.timer.start(33)

    def update_frame(self):
        if self.processing_frame:
            return

        self.processing_frame = True
        try:
            if self.cap and self.cap.isOpened():
                self.performance_graph.increment_frame_count()

                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.timer.stop()
                    return

                self.frame_count += 1
                if self.frame_count % 3 != 0:
                    self.processing_frame = False
                    return

                frame = cv2.resize(frame, (self.video_width, self.video_height))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

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

        cap = cv2.VideoCapture(self.video_path)
        frames = []
        sample_every_n = 5
        count = 0
        
        progress = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            count += 1
            progress.setValue(int(count/total_frames*100))
            if progress.wasCanceled():
                break
                
            if count % sample_every_n != 0:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb)
            
            # Filter faces by minimum size
            faces = [face for face in faces 
                    if (face[2]-face[0]) >= self.min_face_size and (face[1]-face[3]) >= self.min_face_size]
                    
            encodings = face_recognition.face_encodings(rgb, faces)
            
            for (top, right, bottom, left), encoding in zip(faces, encodings):
                face_img = rgb[top:bottom, left:right]
                if face_img.size == 0:
                    continue
                    
                thumb = cv2.resize(face_img, (100, 100))
                qimg = QImage(thumb.data, 100, 100, 3 * 100, QImage.Format_RGB888)
                frames.append((encoding, qimg))

        cap.release()
        progress.close()

        if not frames:
            QMessageBox.information(self, "No Faces", "No faces detected in video.")
            return

        encodings = np.array([f[0] for f in frames])
        clusterer = DBSCAN(metric='euclidean', eps=0.45, min_samples=3)
        labels = clusterer.fit_predict(encodings)
        clusters = {}
        
        for i, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(frames[i])

        if not clusters:
            QMessageBox.information(self, "No Clusters", "No recognizable face groups found to register.")
            return

        for cluster_id, samples in clusters.items():
            # limit to 25 diverse samples
            selected = samples[:25] if len(samples) > 25 else samples
            imgs = [s[1] for s in selected]
            dialog = FaceSelectionDialog(imgs, self)
            if dialog.exec_():
                indices = dialog.get_selected_indices()
                if indices:
                    name, ok = QInputDialog.getText(self, "Register Person", f"Enter name for Cluster {cluster_id}:")
                    if ok and name:
                        self.save_to_database(name, [selected[i][0] for i in indices])
        self.load_registered_faces()

    def process_next_cluster(self):
        if self.current_cluster_index >= len(self.clusters):
            QMessageBox.information(self, "Complete", "All face clusters have been registered.")
            self.clusters.clear()
            return

        current_cluster = self.clusters[self.current_cluster_index]
        dialog = FaceSelectionDialog([f[1] for f in current_cluster], self)
        if dialog.exec_():
            indices = dialog.get_selected_indices()
            if indices:
                name, ok = QInputDialog.getText(self, "Register Person",
                                               f"Enter name for Person {self.current_cluster_index + 1}:")
                if ok and name:
                    selected_encodings = [current_cluster[i][0] for i in indices]
                    self.save_face_to_database(name, selected_encodings)
            self.current_cluster_index += 1
            self.process_next_cluster()
        else:
            self.current_cluster_index += 1
            self.process_next_cluster()

    def save_to_database(self, name, encodings):
        db_path = os.path.join("faceDatabase", f"{name}.pkl")
        data = []
        
        # Load existing encodings if file exists
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                existing_data = pickle.load(f)
                data.extend(existing_data)
                
        # Add new encodings
        data.extend([(name, enc) for enc in encodings])
        
        # Limit to 25 best encodings per person
        if len(data) > 25:
            # Calculate average face distance to other encodings for each encoding
            all_encodings = [enc for _, enc in data]
            avg_distances = []
            for enc in all_encodings:
                distances = face_recognition.face_distance(all_encodings, enc)
                avg_distances.append(np.mean(distances))
            
            # Keep the 25 most representative encodings (lowest average distance)
            best_indices = np.argsort(avg_distances)[:25]
            data = [data[i] for i in best_indices]
        
        with open(db_path, 'wb') as f:
            pickle.dump(data, f)
            
        QMessageBox.information(self, "Saved", f"Saved {len(data)} encodings for '{name}'.")
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            if self.cap:
                self.cap.release()
            self.timer.stop()
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())