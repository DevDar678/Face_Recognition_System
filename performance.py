# performance.py
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time

class PerformanceGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        plt.style.use("dark_background")
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)

        # Layout
        self.layout = QVBoxLayout()
        title_label = QLabel("<h3><b>System Performance Metrics</b></h3>")
        title_label.setStyleSheet("color: white; text-align: center;")
        self.layout.addWidget(title_label)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        # Data initialization
        self.x_data = []
        self.cpu_data = []
        self.mem_data = []
        self.accuracy_data = []
        self.fps_data = []
        self.time_counter = 0
        self.total_faces = 0
        self.correct_matches = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.current_fps = 0

        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)  # Initial call to "prime" it

        # Start with empty data
        for _ in range(20):
            self.x_data.append(0)
            self.cpu_data.append(0)
            self.mem_data.append(0)
            self.accuracy_data.append(0)
            self.fps_data.append(0)

        # Store animation as instance variable
        self.ani = None
        self.start_animation()

    def start_animation(self):
        """Initialize and start the animation"""
        if self.ani is not None:
            self.ani.event_source.stop()
            
        self.ani = animation.FuncAnimation(
            self.figure, 
            self.update_graph, 
            interval=1000, 
            cache_frame_data=False
        )
        self.canvas.draw()

   

    def reset_counters(self):
        """Reset counters when new video is loaded"""
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.current_fps = 0
        self.total_faces = 0
        self.correct_matches = 0

    def increment_frame_count(self):
        """Safe frame counter increment"""
        self.frame_count += 1

    def update_accuracy(self, total_faces, correct_matches):
        """Update face recognition accuracy metrics"""
        self.total_faces += total_faces
        self.correct_matches += correct_matches

    def update_graph(self, frame):
        try:
            

            # Calculate FPS (once per second)
            now = time.time()
            elapsed = now - self.last_fps_update
            if elapsed >= 1.0:
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_update = now

            # System metrics
            cpu_usage = self.process.cpu_percent(interval=None) 


            mem_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Calculate accuracy (handle division by zero)
            accuracy = 0
            if self.total_faces > 0:
                accuracy = (self.correct_matches / self.total_faces) * 100
                # Reset accuracy counters after calculation
                self.total_faces = 0
                self.correct_matches = 0

            # Update data arrays
            self.x_data.append(self.time_counter)
            self.cpu_data.append(cpu_usage)
            self.mem_data.append(mem_usage_mb)
            self.accuracy_data.append(accuracy)
            self.fps_data.append(self.current_fps)

            # Remove oldest data point if we have too many
            if len(self.x_data) > 20:
                self.x_data.pop(0)
                self.cpu_data.pop(0)
                self.mem_data.pop(0)
                self.accuracy_data.pop(0)
                self.fps_data.pop(0)

            # Clear and redraw plot
            self.ax.cla()
            
            # Plot all metrics
            metrics = [
                ('CPU (%)', self.cpu_data, 'r', 'o'),
                ('Memory (MB)', self.mem_data, 'b', 's'),
                ('Accuracy (%)', self.accuracy_data, 'g', '^'),
                ('FPS', self.fps_data, 'y', 'd')
            ]
            
            for label, data, color, marker in metrics:
                line = self.ax.plot(self.x_data, data, f'{color}-', marker=marker, label=label)[0]
                if self.x_data and data:
                    last_x = self.x_data[-1]
                    last_y = data[-1]
                    self.ax.text(last_x, last_y, f"{last_y:.1f}", color=color, fontsize=8)

            # Graph formatting
            self.ax.set_title("Real-Time Performance Metrics", fontsize=12)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Value")
            self.ax.legend(loc="upper right")
            self.ax.grid(True, alpha=0.3)
            
            # Set dynamic Y-axis limits
            max_val = max(max(self.cpu_data), max(self.mem_data), 
                         max(self.accuracy_data), max(self.fps_data), 10)
            self.ax.set_ylim(0, max_val * 1.2)

            self.canvas.draw()
            self.time_counter += 1

        except Exception as e:
            print(f"Graph update error: {str(e)}")
            # Reset on error
            self.reset_counters()

    def closeEvent(self, event):
        """Clean up animation when closing"""
        if self.ani is not None:
            self.ani.event_source.stop()
        super().closeEvent(event)