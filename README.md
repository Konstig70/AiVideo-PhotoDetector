# AI Video and Photo Detector
An AI video and phote detection program that is meant to help users more easily distinguish actual content from AI created ones.

## Why?
With the fast development of GenAI social media platforms are now full of AI generated content, while AI created content is not always bad, it can be used for malicious intent (deepfakes, etc). Our program aims to provide an AI agent that can detect AI created content or check content provided by the user, while also providing easy breakdowns for analyzed content with a relaxed tone.

## How?
We use multi layered checking including human-anatomy detection (hands, sholders, arms and face) and metadata-analysis with some news crosschecking.

## Who would use it?
Our program is meant for both single users and enterprises. For single users it can help them get more reliable information from different medias, for enterprises our software can automate moderating of AI generated conten, if such content is not wanted.

## Roadmap

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
- Descripe what happends in videos with few phareses
- Query news sources (Google API)
- Analyze retrieved articles:
  - Summarize event coverage
  - If no news coverage → increase AI likelihood

**Deliverable:** Fully integrated backend pipeline producing result based on news crosscheck.

---

### Day 5: Justification & Frontend / Demo

**Goal:** Make the tool presentable and demo-ready.

**Tasks:**
- Implement Justification Writing:
  - Combine geometry and metadata
  - Generate human-readable explanation
  - Use OpenAI API for natural language reasoning
- Develop UI / Frontend (Streamlit or Gradio):
  - Upload video → display frame + anomaly overlays
  - Show timeline of anomalies
  - Show final authenticity score + justification
- Test end-to-end pipeline on multiple videos
- Polish demo: make visuals clear, scores interpretable, and flow smooth for judges

**Deliverable:** Fully working MVP demo + exported results / justification.

---

### Also Add
- Option for youtube link
- Export PDF report with overlays, scores, and reasoning



