# PillTrack Edge

![Status](https://img.shields.io/badge/Status-Beta-yellow?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_4%2F5-C51A4A?style=flat-square&logo=raspberrypi)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)
![AWS](https://img.shields.io/badge/Cloud-AWS_S3-232F3E?style=flat-square&logo=amazon-aws)

**PillTrack Edge** is a real-time edge computing application designed for pharmaceutical verification. It utilizes a hybrid computer vision pipeline combining **YOLOv8** for object detection and **DINOv2 (Vision Transformer)** for fine-grained feature extraction to identify pills and verify them against digital prescriptions.

Designed to run efficiently on **ARM-based Edge Devices** (Raspberry Pi) with automated cloud synchronization capabilities.

---

## System Architecture

The system operates on a retrieval-based pipeline:

1.  **Acquisition:** Captures high-res frames via `Picamera2` (XRGB8888) or OpenCV.
2.  **Detection:** YOLOv8 (Nano/Small) localizes pill instances in the frame.
3.  **Embedding:** Cropped instances are processed by **DINOv2 (ViT-B/14)** via ONNX Runtime to generate vector embeddings.
4.  **Identification:** Vectors are compared against a local vector database using Cosine Similarity.
5.  **Verification:** Business logic validates identified pills against the active patient prescription (Mock/HIS).

---

## Key Features

* **Hybrid AI Engine:** Combines the speed of YOLO with the accuracy of Vision Transformers (336x336 input resolution).
* **Smart Cloud Sync:** Automatically checks and downloads model/database updates from **AWS S3** upon boot (Fail-safe architecture).
* **Edge Optimized:** Fully compatible with **ONNX Runtime** on CPU, optimized for low-latency inference on Raspberry Pi.
* **Flexible Config:** Centralized `config.yaml` for adjusting thresholds, model paths, and resolution without code changes.
* **Robust Error Handling:** Includes automated logging, thread-safe processing, and fallback modes for network interruptions.

---

## Installation

### Prerequisites
* Raspberry Pi 4 (4GB+) or Raspberry Pi 5
* Python 3.9 or higher
* AWS Credentials (for S3 Sync)

### 1. Clone the repository
```bash
git clone [https://github.com/your-org/pilltrack-edge.git](https://github.com/your-org/pilltrack-edge.git)
cd pilltrack-edge
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a .env file in the root directory:
```bash
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=ap-southeast-1
```
### 4. Configuration
Modify config.yaml to match your deployment needs:
```bash
artifacts:
  model: 'models/dinov2_vitb14.onnx'
settings:
  yolo_conf: 0.5
  ai_size: 336  # Input resolution for DINOv2
```

### Usage
Run the main orchestrator:

```bash
python main.py
```

## Controls

| Key | Action |
|----|--------|
| **N** | Load the next patient prescription from the queue |
| **T** | Start / Stop the performance stopwatch |
| **Q** | Gracefully stop the application and release camera resources |

## 📁 Project Structure

```text
pilltrack-edge/
├── config.yaml          # Central configuration
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── .env                 # Secrets (Not committed)
├── models/              # ONNX models & weights
├── database/            # Vector DB & drug lists
├── mock_server/         # Mock HIS data
├── scripts/             # Utility scripts (e.g. model export)
└── src/
    ├── ai/              # Inference engines (YOLO, DINO)
    ├── hardware/        # Camera & UI drivers
    ├── services/        # S3 sync & HIS connectors
    ├── core/            # Business logic
    └── utils/           # Helpers & config loaders
```

## 👨‍💻 Author

**Sitta Boonkaew**  
AI Engineer Intern @ AI SmartTech  

---

## 📄 License

© 2025 AI SmartTech. All Rights Reserved.
