# Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

The Driver Drowsiness Detection System is a modular computer-vision application that evaluates eye openness from webcam images. It uses facial landmarks and the Eye Aspect Ratio (EAR) to identify sustained eye closure, display an alert state, record session events, and create downloadable session reports.

Drowsiness can reduce a driver's ability to respond to hazards. This project demonstrates a local, visual monitoring workflow intended for experimentation and educational use. It is not a certified safety device, does not replace rest or driver supervision, and must not be used as the sole basis for safety-critical decisions.

## Features

- Facial landmark-based EAR calculations for both eyes.
- Configurable EAR threshold and consecutive-frame threshold.
- Face rectangle and eye landmark overlays.
- Optional sound alert with a repeat cooldown.
- Timestamped CSV session logs with frame and alert events.
- Session analytics with interactive Plotly visualizations.
- PDF reports with summary statistics, an EAR chart, and an event log.
- Responsive Streamlit dashboard, analytics, reports, and settings pages.

## Tech Stack

| Area | Technologies |
|---|---|
| User interface | Streamlit 1.32.0 |
| Computer vision | OpenCV 4.9.0, dlib 19.24.2 |
| Numeric processing | NumPy 1.26.3, SciPy 1.11.4, Pandas 2.1.4 |
| Charts | Plotly 5.18.0, Matplotlib 3.8.2 |
| Reports and media | ReportLab 4.0.9, Pillow 10.2.0 |
| Testing | pytest 7.4.4 |

## Architecture Diagram

```text
                  ┌──────────────────────────┐
                  │   Streamlit application  │
                  │ Home · Live · Analytics  │
                  │ Reports · Settings       │
                  └────────────┬─────────────┘
                               │ camera image / user settings
                               ▼
                  ┌──────────────────────────┐
                  │   DrowsinessDetector     │
                  │ dlib face + landmarks    │
                  └────────────┬─────────────┘
                               │ eye coordinates
                               ▼
                  ┌──────────────────────────┐
                  │      EAR calculator      │
                  │ frame state + threshold  │
                  └──────┬─────────┬─────────┘
                         │         │
             alert state │         │ event records
                         ▼         ▼
                ┌────────────┐  ┌─────────────────┐
                │ AlertSystem│  │ SessionLogger   │
                └────────────┘  │ CSV session log │
                                └────────┬────────┘
                                         │
                           ┌─────────────┴────────────┐
                           ▼                          ▼
                  ┌────────────────┐         ┌────────────────┐
                  │ Plotly charts  │         │ ReportGenerator│
                  │ analytics view│         │ ReportLab PDF  │
                  └────────────────┘         └────────────────┘
```

## Folder Structure

```text
driver_drowsiness_detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── backend/
│   ├── __init__.py
│   ├── detector.py
│   ├── ear_calculator.py
│   ├── alert_system.py
│   ├── session_logger.py
│   └── report_generator.py
├── frontend/
│   ├── __init__.py
│   ├── components.py
│   ├── styles.py
│   └── charts.py
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── assets/
│   ├── alarm.wav
│   └── logo.png
├── data/
│   └── session_logs/
├── reports/
└── tests/
    ├── test_ear.py
    └── test_detector.py
```

## Installation Steps

1. Install Python 3.10 or later. On Windows, use a Python distribution compatible with the pinned dlib package or install the required C++ build tools.
2. Clone or copy this project and change into its directory:

   ```bash
   cd driver_drowsiness_detection
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install the pinned dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   dlib may require CMake and a C++ compiler when a compatible wheel is unavailable.

5. Verify the included facial landmark model as described below.
6. Start the application:

   ```bash
   streamlit run app.py
   ```

## Download the Dlib Model

The 68-point predictor is distributed by dlib as a compressed model. Download the official archive from:

<https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>

Extract `shape_predictor_68_face_landmarks.dat` into the project's `models/` directory. For example, on macOS/Linux:

```bash
python -c "import bz2,pathlib; p=pathlib.Path('models/shape_predictor_68_face_landmarks.dat.bz2'); pathlib.Path('models/shape_predictor_68_face_landmarks.dat').write_bytes(bz2.decompress(p.read_bytes()))"
```

The extracted predictor is included at `models/shape_predictor_68_face_landmarks.dat`. If it is missing, use the download and extraction steps above. On Windows, use a tool that can extract `.bz2` files or the equivalent Python command after placing the downloaded archive in `models/`. Review the upstream model distribution terms before redistributing it. The application reports a clear model status if the predictor is missing.

## Usage Guide

Run the app locally with `streamlit run app.py`, then open the local URL printed by Streamlit.

- **Home / Dashboard:** Review the project introduction and model readiness.
- **Live Detection:** Configure threshold, consecutive frames, and alarm sound in the sidebar. Start a session, grant camera permission, and capture images for analysis. Stop detection to save the CSV. This implementation uses Streamlit's `st.camera_input`, which analyzes captured images rather than streaming every webcam frame continuously.
- **Analytics Dashboard:** Choose a saved session to view duration, alert count, mean EAR, drowsiness percentage, interactive charts, and its event table.
- **Reports:** Select a saved session and generate a PDF report. Download reports from the page.
- **Settings:** Adjust detector thresholds, review the EAR formula and model status, and reset saved CSV/PDF files after confirmation.

Session CSV files are written to `data/session_logs/`; generated reports are written to `reports/`. These are local files in a local run and ephemeral files on many hosted Streamlit environments.

## Deployment on Streamlit Cloud

1. Push the project to a GitHub repository.
2. Keep `app.py` at the repository root for the deployed app, or select `driver_drowsiness_detection/app.py` if the project is in a subdirectory.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in, and select **Create app**.
4. Choose the repository, branch, and `app.py` entry point.
5. Deploy and inspect build logs if dependency installation fails.
6. Ensure the predictor model is available at `models/shape_predictor_68_face_landmarks.dat` in the deployment environment. Do not commit credentials or secrets.
7. Test camera permissions, model initialization, report creation, and storage behavior in the deployed environment.



The pinned dlib dependency can be difficult to build on hosted Linux systems and may require a compatible Python/runtime image. Webcam access and writable disk storage also depend on the browser and hosting environment. Streamlit Cloud's filesystem may be reset, and `st.camera_input` is capture-based, not a continuous low-latency video feed.

## Project Workflow

The detector locates a face and the dlib predictor estimates 68 facial landmarks. Each eye is represented by six ordered points. The EAR is computed as:

```text
           ||p2 - p6|| + ||p3 - p5||
EAR =      -------------------------
                   2 ||p1 - p4||
```

The two eye EAR values are averaged. A value below the configured threshold increments the closed-eye frame counter; an open-eye result resets it. Once the counter reaches the configured consecutive-frame count, the state becomes drowsy and an alert event is logged. EAR behavior varies with subject, camera placement, face orientation, lighting, and landmark quality, so thresholds should be calibrated and should not be treated as medical or safety certification.


## Troubleshooting

| Issue | Suggested fix |
|---|---|
| Model status says missing | Extract the `.bz2` archive and verify the `.dat` file is directly in `models/`. |
| dlib installation fails | Install a supported compiler/CMake toolchain or use a Python environment with a compatible prebuilt dlib wheel. |
| Camera is unavailable | Use a browser with camera permission, use HTTPS for remote access, and close applications that hold the webcam. |
| No face is detected | Improve front lighting, face the camera, reduce occlusion, and move closer to the camera. |
| Alarm is silent | Check the alarm file and system output device. Linux playback requires `aplay`; macOS uses `afplay`; browser/server audio behavior varies by deployment. |
| PDF generation fails | Confirm ReportLab and Matplotlib are installed and that `reports/` is writable. |
| No session in analytics | Start a session, capture at least one frame, and stop the session so its CSV is saved. |
| Hosted deployment fails at dlib build | Review build logs and use a Python version/platform for which a compatible dlib build is available. |

## Future Enhancements

- Continuous webcam streaming with browser-side video frames.
- Calibrated thresholds per driver and camera configuration.
- More robust pose and lighting compensation.
- Optional local database and configurable retention policy.
- Container-based deployment with a reproducible dlib build.
- Additional validation on diverse lighting, camera, and subject conditions.

## Author, License, and Acknowledgments

Created as a modular full-stack computer-vision project. Licensed under the MIT License; see [LICENSE](LICENSE). Acknowledgments go to the dlib project for facial landmark tooling, the OpenCV project for computer-vision primitives, and the Streamlit, SciPy, Plotly, Matplotlib, and ReportLab communities for the supporting frameworks.
