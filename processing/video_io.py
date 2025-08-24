import cv2
import os

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames

def save_video(frames, output_path, fps=120):
    if not frames:
        raise ValueError("No frames to save.")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
