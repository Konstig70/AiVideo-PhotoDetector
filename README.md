# AI Video and Photo Detector
An AI video and phote detection program that is meant to help users more easily distinguish actual content from AI created ones. 

## Why?
With the fast development of GenAI social media platforms are now full of AI generated content, while AI created content is not always bad, it can be used for malicious intent (deepfakes, etc). Our program aims to provide an AI agent that can detect AI created content or check content provided by the user, while also providing easy breakdowns for analyzed content with a relaxed tone.

## How?
We use multi layered checking including geometric-, shape-, basic objectdetection- and metadata-analysis with some news crosschecking.

## Roadmap

### Phase 0: Preparation (Before Day 1)

#### Environment & Tools
- Install Python ≥3.10
- Install VSCode + Jupyter extension
- Optional: PyCharm for debugging
- Create virtual environment or Anaconda environment
- Install required libraries:
  - opencv-python
  - mediapipe
  - numpy
  - matplotlib
  - pandas
  - scikit-learn
  - streamlit
  - gradio
  - pymediainfo
  - requests
  - openai

---

### Day 1: Geometry Detection Prototype

**Goal:** Get a working geometry detection pipeline on sample videos.

**Tasks:**
- Experiment in Jupyter:
  - Load sample videos → extract frames (OpenCV)
  - Detect human landmarks (MediaPipe: hands, pose, face)
  - Extract object contours (OpenCV)
  - Overlay landmarks and contours for visualization
- Test basic geometry plausibility checks:
  - Hands: 5 fingers, angles
  - Pose: limb ratios
  - Face: symmetry
  - Object contours: smooth/closed polygons
- Output: simple anomaly flag per frame

**Deliverable:** Prototype notebook with visual results on 1–2 videos.

---

### Day 2: Temporal Consistency & Scoring

**Goal:** Aggregate per-frame anomalies into video-level AI likelihood.

**Tasks:**
- Implement temporal consistency logic:
  - Sliding window / streak detection
  - Count consecutive anomalous frames → raise likelihood
- Calculate frame-level anomaly scores → normalized video score
- Integrate basic visualization:
  - Timeline of anomalies
  - Highlight frames with issues
- Refactor working functions from Jupyter → VSCode modules:
  - `geometry_mapping/anomaly_scoring.py`
  - `geometry_mapping/temporal_analysis.py`

**Deliverable:** Video-level anomaly scoring + visual timeline ready for demo.

---

### Day 3: Metadata Scrutiny

**Goal:** Extract and analyze video metadata for signs of AI generation.

**Tasks:**
- Extract metadata using ffmpeg / pymediainfo:
  - Resolution, codec, timestamps, device info, editing traces
- Implement plausibility checks:
  - Missing device info → suspicious
  - Strange encoding or timestamps → flag
- Aggregate metadata anomalies → frame/video-level scores
- Visualization: simple table or overlay showing flagged metadata

**Deliverable:** Metadata module integrated into pipeline + score ready to combine with geometry score.

---

### Day 4: News Crosscheck & Integration

**Goal:** Cross-reference video events with real-world news and combine all signals.

**Tasks:**
- Extract event info from video title/description or keywords
- Query news sources (Bing, Google, Twitter API)
- Analyze retrieved articles:
  - Summarize event coverage
  - If no news coverage → increase AI likelihood
- Combine geometry score + metadata score + news score → overall authenticity score
- Implement scoring normalization and weighting

**Deliverable:** Fully integrated backend pipeline producing final authenticity score for videos.

---

### Day 5: Justification & Frontend / Demo

**Goal:** Make the tool presentable and demo-ready.

**Tasks:**
- Implement Justification Writing:
  - Combine geometry, metadata, and news signals
  - Generate human-readable explanation
  - Optionally use OpenAI API for natural language reasoning
- Develop UI / Frontend (Streamlit or Gradio):
  - Upload video → display frame + anomaly overlays
  - Show timeline of anomalies
  - Show final authenticity score + justification
- Test end-to-end pipeline on multiple videos
- Polish demo: make visuals clear, scores interpretable, and flow smooth for judges

**Deliverable:** Fully working MVP demo + exported results / justification.

---

### Optional Enhancements (If Time Permits)
- Lightweight ML anomaly detection on geometry features (Isolation Forest / PCA)
- Batch video processing for multiple files
- Export PDF report with overlays, scores, and reasoning



