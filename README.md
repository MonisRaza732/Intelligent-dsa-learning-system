# 🧠 DSA Intelligence — Big Data Learning System

An ML-powered web application that analyzes your LeetCode profile, predicts your skill level, and recommends company-specific questions. Built using **Apache Spark (PySpark MLlib)** and **MongoDB** for large-scale data processing and persistence.

## Features

- **LeetCode Profile Analysis** — Fetches data via GraphQL (problem stats, contest info, topic skills)
- **ML Skill Prediction** — Random Forest classifier predicts skill tier (Beginner → Expert)
- **Placement Readiness Score** — Gradient Boosting regressor scores placement readiness (0–100)
- **Topic-Wise Breakdown** — Identifies strengths and areas to improve
- **Company Recommendations** — Questions from Google, Amazon, Microsoft, Meta, Apple, Goldman Sachs, Uber, Adobe
- **Personalized Study Plan** — Daily targets, weekly schedule, and milestones

## Tech Stack

| Layer      | Technology                   |
|------------|------------------------------|
| Backend    | Flask, Flask-CORS            |
| Frontend   | HTML5, CSS3, Vanilla JS      |
| Big Data ML| PySpark MLlib (Spark 3.5+)   |
| Database   | MongoDB + pymongo            |
| Data       | LeetCode GraphQL API         |

## Project Structure

```
DSA project/
├── app.py                  # Flask application (API routes)
├── spark_ml.py             # PySpark MLlib model training & prediction
├── mongo_handler.py        # MongoDB data persistence layer
├── data_pipeline.py        # Spark batch processing pipeline
├── ml_model.py             # scikit-learn fallback ML models
├── leetcode_fetcher.py     # LeetCode GraphQL data fetcher
├── recommender.py          # Question recommender + study plan
├── requirements.txt        # Python dependencies
├── data/
│   └── company_questions.json  # Company-specific question database
├── models/                 # Auto-generated trained ML models
├── static/
│   ├── css/
│   │   └── style.css       # Premium dark theme styles
│   └── js/
│       └── app.js          # Frontend application logic
└── templates/
    └── index.html          # Main HTML template
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Java 8 or 11 (Required for Apache Spark)
- MongoDB running locally on port 27017 (Optional, uses in-memory fallback if not found)

*Note: The system features graceful degradation. If Java/PySpark are not installed, it falls back to scikit-learn. If MongoDB is not running, it falls back to an in-memory dictionary.*

### 1. Create Virtual Environment (Recommended)

```bash
cd "DSA project"
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The app starts at **http://127.0.0.1:5000**

### 4. Use the App

1. Open **http://127.0.0.1:5000** in your browser
2. Enter a LeetCode username (e.g., `neal_wu`, `tourist`)
3. Click **Analyze** to see your full profile analysis

## API Endpoints

| Method | Endpoint          | Body                              | Description                           |
|--------|-------------------|------------------------------------|---------------------------------------|
| GET    | `/`               | —                                  | Serves the frontend                   |
| POST   | `/api/analyze`    | `{ "username": "..." }`           | Full profile analysis + ML prediction |
| POST   | `/api/recommend`  | `{ "username": "...", "company": "Google" }` | Company-specific recommendations |
| GET    | `/api/companies`  | —                                  | List supported companies              |

## How the Big Data Pipeline Works

1. **Synthetic Training Data** — `data_pipeline.py` uses Spark to generate **100,000** realistic user profiles across 4 skill tiers.
2. **Distributed Feature Engineering** — Uses PySpark `VectorAssembler` and `StandardScaler` to extract 10 features natively in the Spark engine.
3. **PySpark MLlib Models**:
   - `RandomForestClassifier` → Skill level prediction
   - `GBTRegressor` → Placement readiness scoring
4. **Data Persistence** — Stores all analysis results, ML runs, and pipeline reports in **MongoDB** (`mongo_handler.py`).
