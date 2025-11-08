# Video Feature Extraction Tool

This project is a **Python-based Computer Vision tool** I built to extract meaningful context from a video — including scene changes, motion intensity, on-screen text, and object/person presence.

It’s implemented using **OpenCV**, **pytesseract (OCR)**, and **YOLOv3-tiny (object detection)** inside a single modular class: `VideoAnalyzer`.

---

## Core Design

The main class, `VideoAnalyzer`, handles all steps:
1. Load the video with `cv2.VideoCapture`.
2. Sample frames for analysis.
3. Extract four key visual features.
4. Save the final results to JSON.

Each analysis step is implemented in a dedicated method for clarity and reusability.

---

## Key Features

### 1. Shot Cut Detection — Scene Change Recognition  
**Goal:** Detect sudden scene transitions (“hard cuts”).  
**Used:** `cv2.calcHist()`, `cv2.compareHist()`  

In `_extract_shot_cuts()`, each frame’s grayscale histogram is compared with the previous frame:

```python
if cv2.compareHist(hist_prev, hist_current, cv2.HISTCMP_CORREL) < 0.5:
    shot_cuts += 1
```

When the correlation drops below `0.5`, it indicates a strong visual difference → counted as a scene cut.  
This helps identify where one scene ends and another begins.

---

### 2. Motion Analysis — Quantifying Movement  
**Goal:** Measure average motion in the video (calm vs. high action).  
**Used:** `cv2.calcOpticalFlowFarneback()`  

In `_extract_motion()`, I use **dense optical flow** to compute pixel-level movement between consecutive frames:

```python
flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

The mean magnitude of flow vectors gives a **motion score**.  
Higher means more activity, lower means static scenes.

---

### 3. Text Detection (OCR) — Reading On-Screen Words  
**Goal:** Extract text content and measure how often text appears.  
**Used:** `pytesseract`, `OpenCV` thresholding, `regex`, `Counter`  

In `_extract_text_features()`, frames are preprocessed and passed to Tesseract OCR:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text = pytesseract.image_to_string(thresh, timeout=3).strip()
```

Detected text is aggregated across all frames.  
I then compute:
- **`text_present_ratio`** → fraction of frames containing text  
- **`keywords`** → top 20 frequent words using regex + `Counter`

```python
keywords = [w for w, _ in Counter(re.findall(r'\b[a-z]{4,}\b', ' '.join(all_text).lower())).most_common(20)]
```

---

### 4. Object vs. Person Dominance — Visual Composition  
**Goal:** Determine whether people or other objects dominate the video.  
**Used:** `YOLOv3-tiny`, `cv2.dnn` module  

I load the YOLOv3-tiny model (downloaded separately) and run inference in `_extract_object_features()`:

```python
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
self.net.setInput(blob)
layer_outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
```

For each detection:
- If `label == "person"` → count as person frame  
- Otherwise → count as object frame  

The ratio gives an idea of human vs. non-human visual focus.

---

## Processing Pipeline

The main execution happens in `run_analysis()`:

```python
frames, frames_gray = [], []
while True:
    ret, frame = self.cap.read()
    if not ret: break
    frames.append(frame)
    frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
```

Steps executed sequentially:
1. **Frame Sampling** — extract every N-th frame for efficiency.  
2. **Shot Cut Detection** — `_extract_shot_cuts(frames_gray)`  
3. **Motion Analysis** — `_extract_motion(frames_gray)`  
4. **Text Extraction** — `_extract_text_features(frames)`  
5. **Object Analysis** — `_extract_object_features(frames)`  
6. **Results Aggregation** — combine everything into one dictionary.

---

## Output

Final output (JSON format):

```json
{
  "filename": "OpenCV-Tutorial.mp4",
  "duration_seconds": 0.0,
  "shot_cuts": 2,
  "avg_motion_score": 2.6984105110168457,
  "text_present_ratio": 0.9854651162790697,
  "detected_text_keywords": [
    "image",
    "this",
    "....."
  ],
  "person_object_ratio": 0.5555555555555556,
  "processing_time_seconds": 396.4841549396515
}
```

Saved automatically as:  
`<video_name>_features.json`

---

## Libraries & Models Used

| Component | Purpose |
|------------|----------|
| **OpenCV (cv2)** | Frame capture, histogram, optical flow, DNN model loading |
| **pytesseract** | Optical Character Recognition (OCR) |
| **YOLOv3-tiny** | Object and person detection |
| **NumPy** | Array & math operations |
| **regex, Counter** | Keyword extraction |
| **JSON** | Exporting structured output |

---

## Summary

This tool efficiently converts raw video into structured, human-understandable context by combining:
- **Scene-level analysis** (shot cuts, motion)
- **Content-level analysis** (text, objects, people)
- **Modular design** for easy future extension (e.g., face detection, audio analysis)

Each method in the `VideoAnalyzer` class reflects a focused and independent vision feature — designed for clarity, explainability, and reliability.

---
