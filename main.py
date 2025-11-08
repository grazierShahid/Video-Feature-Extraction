import cv2
import numpy as np
import pytesseract
import json
import argparse
import time
import re
from pathlib import Path
from collections import Counter

class VideoAnalyzer:
    """
    Analyzes video files to extract features.
    The analysis is structured sequentially for clarity while maintaining performance.
    """
    def __init__(self, video_path: str):
        # Initialize video capture and get all properties upfront
        self.video_path = Path(video_path)
        if not self.video_path.is_file(): raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened(): raise ValueError(f"Cannot open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0

        # Load the pre-trained YOLOv3-tiny model for object detection
        yolo_dir = Path("yolo_model")
        weights = yolo_dir / "yolov3-tiny.weights"
        cfg = yolo_dir / "yolov3-tiny.cfg"
        names = yolo_dir / "coco.names"
        if not all([weights.is_file(), cfg.is_file(), names.is_file()]):
            raise FileNotFoundError(f"YOLO model files not found in {yolo_dir}.")
        
        self.net = cv2.dnn.readNet(str(weights), str(cfg))
        with open(names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def _extract_shot_cuts(self, frames_gray: list) -> int:
        """1. Analyzes a list of grayscale frames for hard cuts."""
        shot_cuts = 0
        for i in range(1, len(frames_gray)):
            # Compare the histogram of the current frame to the previous one
            hist_prev = cv2.calcHist([frames_gray[i-1]], [0], None, [256], [0, 256])
            hist_current = cv2.calcHist([frames_gray[i]], [0], None, [256], [0, 256])
            cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_current, hist_current, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # A low correlation score indicates a high difference, hence a shot cut
            if cv2.compareHist(hist_prev, hist_current, cv2.HISTCMP_CORREL) < 0.5:
                shot_cuts += 1
        return shot_cuts

    def _extract_motion(self, frames_gray: list) -> float:
        """2. Analyzes a list of grayscale frames for motion."""
        motion_scores = []
        for i in range(1, len(frames_gray)):
            # Calculate dense optical flow to estimate motion between frames
            flow = cv2.calcOpticalFlowFarneback(frames_gray[i-1], frames_gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(np.mean(magnitude))
        return np.mean(motion_scores) if motion_scores else 0.0

    def _extract_text_features(self, frames: list) -> (float, list):
        """3. Analyzes a list of color frames for text content."""
        all_text = []
        for frame in frames:
            try:
                # Use automatic thresholding and OCR to find text in the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(thresh, timeout=3).strip()
                if text:
                    all_text.append(text)
            except (RuntimeError, pytesseract.TesseractError):
                continue # Ignore frames where OCR fails or times out
        
        # Calculate the ratio of frames where text was found
        text_ratio = len(all_text) / len(frames) if frames else 0.0
        
        # Find the top 10 most common words (keywords) from all detected text
        keywords = [w for w, _ in Counter(re.findall(r'\b[a-z]{4,}\b', ' '.join(all_text).lower())).most_common(10)]
        return text_ratio, keywords

    def _extract_object_features(self, frames: list) -> (int, int):
        """4. Analyzes a list of color frames for people and objects using YOLO."""
        person_frames, object_frames = 0, 0
        for frame in frames:
            # Prepare the frame and run it through the YOLO network
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            
            has_person, has_object = False, False
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    # Check if the detection confidence is above a threshold
                    if scores[np.argmax(scores)] > 0.5:
                        label = self.classes[np.argmax(scores)]
                        if label == "person": has_person = True
                        else: has_object = True
            
            # Count frames containing at least one person or object
            if has_person: person_frames += 1
            if has_object: object_frames += 1
        return person_frames, object_frames

    def run_analysis(self):
        """
        Executes the full analysis by calling each feature method sequentially.
        """
        start_time = time.time()
        
        # This is the only loop that reads the video file, making it efficient.
        # We store a sample of frames in memory to pass to the analysis methods.
        print("Step 1: Sampling frames from video...")
        frames, frames_gray = [], []
        sample_rate = 1
        frame_interval = int(self.fps / sample_rate) if sample_rate > 0 else 1
        
        frame_number = 0
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if not ret: break
            frames.append(frame)
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame_number += frame_interval
        self.cap.release()

        # Each method is called one-by-one on the sampled frames for clarity.
        print("Step 2: Detecting shot cuts...")
        shot_cuts = self._extract_shot_cuts(frames_gray)
        
        print("Step 3: Analyzing motion...")
        avg_motion = self._extract_motion(frames_gray)
        
        print("Step 4: Extracting text and keywords...")
        text_ratio, keywords = self._extract_text_features(frames)
        
        print("Step 5: Detecting people and objects...")
        person_frames, object_frames = self._extract_object_features(frames)

        # Collate all the extracted data into a final dictionary.
        print("Step 6: Finalizing results...")
        total_entities = person_frames + object_frames
        
        return {
            "filename": self.video_path.name,
            "duration_seconds": self.duration,
            "shot_cuts": shot_cuts,
            "avg_motion_score": avg_motion,
            "text_present_ratio": text_ratio,
            "detected_text_keywords": keywords,
            "person_object_ratio": person_frames / total_entities if total_entities > 0 else 0.0,
            "processing_time_seconds": time.time() - start_time
        }

def save_and_print_results(features: dict, video_path: Path):
    """Saves features to a JSON file and prints them to the console."""
    if not features:
        print("Could not extract features.")
        return

    # Define an output path for the JSON file based on the video's name
    output_path = video_path.parent / f"{video_path.stem}_features.json"
    
    # Custom JSON encoder to handle special numpy data types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)): return float(obj)
            return super(NumpyEncoder, self).default(obj)

    # Write the dictionary to a JSON file
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2, cls=NumpyEncoder)

    # Print the results to the console for immediate feedback
    print(f"\nAnalysis complete. Results saved to: {output_path}")
    print("\n--- Feature Summary ---")
    print(json.dumps(features, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    # This is the main entry point when the script is run from the command line.
    parser = argparse.ArgumentParser(description="Video Feature Extraction Tool")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    args = parser.parse_args()

    try:
        # Create an analyzer instance and run the analysis
        analyzer = VideoAnalyzer(args.video)
        extracted_features = analyzer.run_analysis()
        save_and_print_results(extracted_features, Path(args.video))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")