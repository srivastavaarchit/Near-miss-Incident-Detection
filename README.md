# 🚨 Near-Miss Chain Snatching Incident Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange?style=for-the-badge&logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle)

**An AI-powered CCTV surveillance system that catches chain snatching attempts — even the ones that almost happened.**

[Overview](#-overview) • [How It Works](#-how-it-works) • [Tech Stack](#-tech-stack) • [Results](#-results) • [Getting Started](#-getting-started) • [Outputs](#-outputs)

</div>

---

## 🔍 Overview

Chain snatching is a fast, opportunistic crime that often goes undetected by traditional surveillance systems — especially failed attempts. This project tackles a critical gap in public safety: **detecting near-miss incidents** where a suspect approaches a pedestrian, attempts to snatch a chain, but fails due to the victim's reaction or external interruption.

By analyzing CCTV footage using a multi-condition AI pipeline, this system identifies suspicious patterns **before** a crime is completed, enabling law enforcement to:

- 🗺️ **Map high-risk zones** before incidents escalate
- ⚠️ **Generate early warnings** for patrol deployment
- 📊 **Build evidence trails** from failed attempts
- 🔮 **Predict and prevent** future crimes using historical patterns

> *If we can see a failed attempt, we can stop the next one.*

---

## 🧠 How It Works

The detection engine runs a **6-stage multi-condition analysis pipeline** on each video frame, assigning a confidence score based on how many risk conditions are simultaneously satisfied.

```
CCTV Feed → Object Detection → Distance Analysis → Motion Analysis
         → Timing Filter → Reaction Detection → Confidence Scoring → Alert
```

### Stage 1 — Object Detection (YOLOv11)
Uses `yolo11n.pt` to detect and track pedestrians (victims) and vehicles (suspects — bikes, motorbikes, cars) across frames with persistent Track IDs.

### Stage 2 — Distance Analysis
Computes real-world distance (in meters) between the nearest pedestrian and approaching vehicle. Triggers suspicion when proximity drops below **1.8 meters**.

### Stage 3 — Motion Analysis
Detects sudden, fast hand movements directed toward the upper-body / neck zone of the pedestrian — the hallmark motion of a snatching attempt.

### Stage 4 — Timing Filter
Flags interactions lasting between **0.5 – 6.0 seconds**, filtering out both incidental passing and prolonged normal conversations.

### Stage 5 — Victim Reaction Detection
Identifies defensive victim responses: stepping back, sudden turns, running, or abrupt posture changes — behavioral signals of an attempted grab.

### Stage 6 — Confidence Scoring
Combines all conditions into a weighted score. An alert fires only when **≥ 5 of 7 conditions** are met with an overall confidence **≥ 50%**, drastically reducing false positives.

---

## ⚙️ Detection Configuration

| Parameter | Value | Description |
|---|---|---|
| `dist_threshold_m` | 1.8 m | Minimum suspect–victim proximity |
| `pixels_per_meter` | 60 px | Scene calibration factor |
| `motion_speed_threshold` | 0.08 | Minimum hand speed ratio |
| `interaction_min_sec` | 0.5 s | Minimum interaction window |
| `interaction_max_sec` | 6.0 s | Maximum interaction window |
| `min_conditions` | 5 / 7 | Conditions required for alert |
| `min_confidence` | 50% | Score threshold for alert |
| `alert_cooldown_sec` | 5.0 s | Prevents duplicate alerts |
| `raw_score_threshold` | 0.62 | Fine-grained alert gate |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11 (Ultralytics) |
| Tracking | ByteTrack (via Ultralytics) |
| Video Processing | OpenCV (`cv2`) |
| Numerical Analysis | NumPy, Pandas |
| Model Checkpoint | PyTorch (`.pth`) |
| Visualization | Matplotlib |
| Platform | Kaggle Notebooks |

---

## 📂 Project Structure

```
📦 near-miss-detector/
├── 📓 notebook48faa3d2e6.ipynb     # Main detection pipeline
├── 📁 kaggle/working/
│   ├── near_miss_results.csv       # Frame-level detection log
│   ├── near_miss_summary.csv       # Per-video summary
│   ├── model_accuracy_report.csv   # TP/TN/FP/FN metrics
│   ├── near_miss_model.pth         # Lightweight checkpoint
│   ├── near_miss_model_full.pth    # Full checkpoint with YOLO weights
│   ├── near_miss_dashboard.png     # Confidence & distance charts
│   ├── normal_video_results.png    # False-positive analysis chart
│   └── output_video_N.mp4/.avi     # Annotated output videos
└── 📄 README.md
```

---

## 📊 Results & Outputs

The pipeline generates a full evaluation suite after each run:

### 🎯 Annotated Video Output
Each processed video is annotated in real-time with:
- 🟥 **VICTIM** bounding box (red) with Track ID
- 🟧 **SUSPECT VEHICLE** bounding box (orange) with type label
- 📏 **Distance line** between victim and suspect (color-coded by proximity)
- 🚨 **"NEAR-MISS / SNATCHING DETECTED"** banner with live confidence %
- ⚠️ **"SUSPICIOUS ACTIVITY"** banner for borderline events

### 📈 Analytics Dashboard
- Confidence score over all frames
- Victim–suspect distance timeline with threshold overlay
- Conditions met distribution histogram
- False-positive analysis for normal videos

### 📋 CSV Reports
- **Frame-level log**: timestamp, distance, motion speed, conditions met, confidence, alert flag
- **Per-video summary**: total alerts, max confidence, ground truth vs prediction
- **Accuracy report**: Precision, Recall, F1, Accuracy, TP/TN/FP/FN

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics opencv-python-headless numpy pandas torch matplotlib scikit-learn
```

### Run on Kaggle

1. Fork the notebook on Kaggle
2. Add the datasets:
   - `tharunravilla/near-miss-incident1` (near-miss videos)
   - `tharunravilla/normal-videos-micro` (normal videos for evaluation)
3. Enable GPU accelerator (T4 recommended)
4. **Run All** — the pipeline processes all videos and exports results automatically

### Load the Saved Checkpoint

```python
import torch

def load_near_miss_checkpoint(pth_path: str) -> dict:
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    assert ckpt.get("model_name") == "NearMissDetector"
    return ckpt

ckpt = load_near_miss_checkpoint("near_miss_model.pth")

# Access thresholds, config, and metrics instantly
print(ckpt["inference_thresholds"])
print(ckpt["metrics"])  # accuracy, f1, TP, TN, FP, FN
```

---

## 🎬 Dataset

| Split | Videos | Label |
|---|---|---|
| Near-Miss | 10 | `1` (positive) |
| Normal | Collected via `tharunravilla/normal-videos-micro` | `0` (negative) |

Videos include real CCTV footage of chain snatching scenarios, sourced and labeled for binary classification.

---

## ⚡ Key Challenges & Solutions

| Challenge | Solution |
|---|---|
| Normal movement vs. real attempt | Multi-condition gating (5/7 conditions required) |
| High false positive rate | Suspect & victim minimum track frame requirements |
| Lack of labeled training data | Confidence-score system with simulated + real data |
| Camera angle variation | Aspect ratio filters + spatial zone analysis |
| Duplicate alerts on the same event | 5-second cooldown between successive alerts |

---

## 🔮 Future Improvements

- [ ] Real-time streaming support (RTSP / WebSocket)
- [ ] GPU-optimized inference for edge deployment
- [ ] Multi-camera tracking with scene stitching
- [ ] Dashboard UI for security operators
- [ ] Integration with police alert systems via API

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or raise an issue.

---

## 📄 License

This project is for academic and research purposes. Usage of CCTV footage must comply with local privacy laws and regulations.

---

<div align="center">

Built with 🔍 vigilance and 🤖 intelligence to make public spaces safer.

**If you found this useful, give it a ⭐ on GitHub!**

</div>
