# main.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from singleFaceRec import VideoPlayer as SingleFaceVideoPlayer
from parallel import VideoPlayer as ParallelVideoPlayer

class ComparisonWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single vs Parallel Face Recognition Comparison")
        self.setStyleSheet("background-color: #2E2E2E;")
        self.setGeometry(10, 40, 2200, 900)

        # Main vertical layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("<h1>Face Recognition Systems Comparison</h1>")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
        QLabel {
        font-size: 24px;
        font-weight: bold;
        padding: 15px;
        color: #FFFFFF;
        border-bottom: 2px solid #444;
        }
        """)
        main_layout.addWidget(title)

        # Horizontal layout for both video players
        players_layout = QHBoxLayout()

        # Single Face Recognition (Without Parallelization)
        self.single_face_player = SingleFaceVideoPlayer()
        self.single_face_player.setWindowTitle("SingleFaceRec - No Parallelization")
        players_layout.addWidget(self.single_face_player)

        # Parallelized Face Recognition
        self.parallel_face_player = ParallelVideoPlayer()
        self.parallel_face_player.setWindowTitle("Parallel - With Parallelization")
        players_layout.addWidget(self.parallel_face_player)

        # Add the horizontal layout into the vertical layout
        main_layout.addLayout(players_layout)

        # Set main layout
        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ComparisonWindow()
    window.show()
    sys.exit(app.exec_())
