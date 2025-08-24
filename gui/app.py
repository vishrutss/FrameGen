import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from processing.video_io import extract_frames, save_video
from processing.frame_interpolator import interpolate_frames

class FrameApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Frame Interpolator")
        self.setGeometry(200, 200, 400, 200)
        layout = QVBoxLayout()

        self.label = QLabel("Upload a video to increase FPS")
        layout.addWidget(self.label)

        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_btn)

        self.setLayout(layout)

    def upload_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file:
            self.label.setText("Processing...")
            frames = extract_frames(file, None)
            new_frames = interpolate_frames(frames, target_fps=120, original_fps=30)
            save_video(new_frames, "./data/output/output.mp4", fps=120)
            self.label.setText("Done! Saved to ./data/output/output.mp4")

def run_app():
    app = QApplication(sys.argv)
    window = FrameApp()
    window.show()
    sys.exit(app.exec_())
