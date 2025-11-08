# Video Feature Extraction Tool

A simple, fast, and accurate Python tool to extract key features from video files. It processes videos in a single pass for high performance and saves the results to a JSON file.

## Installation in 4 Steps

### 1. Clone the Repository
First, get the code by cloning the repository to your local machine.
```bash
git clone https://github.com/grazierShahid/Video-Feature-Extraction.git
cd Video-Feature-Extraction
```

### 2. Install Python Packages
Open your terminal in the project folder and run:
```bash
pip install opencv-python pytesseract numpy
```

### 3. Install Tesseract OCR Engine
This is required for text detection.

-   **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get install tesseract-ocr
    ```
-   **macOS (Homebrew):**
    ```bash
    brew install tesseract
    ```
-   **Windows:**
    Download and run the Tesseract installer from the [UB-Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to add its location to your system's PATH.

### 4. Download YOLOv3-tiny Model
This is required for object detection. Run this command in your terminal to download the model into a `yolo_model` folder.
```bash
mkdir -p yolo_model && \
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O yolo_model/yolov3-tiny.weights && \
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -O yolo_model/yolov3-tiny.cfg && \
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O yolo_model/coco.names
```
*(Note: If `wget` is not available, open the URLs in a browser and save the files manually into a `yolo_model` directory.)*

## Usage
Run the script from your terminal, providing the path to your video.
```bash
python main.py --video /path/video.mp4
```
The script will create a JSON file with the results in the same directory as your video.

### Example
```bash
python main.py --video resources/OpenCV-Tutorial.mp4
```
This creates a results file named `resources/OpenCV-Tutorial_features.json`.
