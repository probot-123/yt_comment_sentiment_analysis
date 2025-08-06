# YouTube Comment Sentiment Analysis

A data science project and Chrome extension to detect sentiment in YouTube comments. This project includes a Flask API for sentiment analysis, machine learning models, and a browser extension for real-time feedback.

## Features

- Chrome extension for YouTube comment sentiment detection
- Flask API for serving sentiment predictions
- Machine learning pipeline for training and evaluating models
- Reproducible experiments with DVC
- Docker support for easy deployment

## Project Structure

```
├── flask_app/           # Flask API app
├── src/                 # Source code (data, features, models, visualization)
├── data/                # Data storage (raw, processed, etc.)
├── models/              # Trained models and outputs
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/             # Utility and test scripts
├── deploy/              # Deployment scripts (Docker, etc.)
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
├── dvc.yaml             # DVC pipeline definition
├── Dockerfile           # Docker setup
└── README.md            # Project overview (this file)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd yt_comment_sentiment_analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   cd flask_app && pip install -r requirements.txt
   ```
3. **(Optional) Set up with Docker:**
   ```bash
   docker build -t yt-sentiment .
   docker run -p 5000:5000 yt-sentiment
   ```

## Usage

- **Run Flask API locally:**
  ```bash
  cd flask_app
  python app.py
  ```
- **Test API:**
  ```bash
  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"comment": "This video is awesome!"}'
  ```
- **Use Chrome Extension:**
  1. Load the extension from the Chrome extensions page (`chrome://extensions` > Load unpacked).
  2. Interact with YouTube comments to see sentiment predictions.

## Reproducibility

- Data and model pipeline managed with [DVC](https://dvc.org/).
- Notebooks and scripts for data processing and model training in `src/` and `notebooks/`.


_Project inspired by the [cookiecutter data science template](https://drivendata.github.io/cookiecutter-data-science/)._
