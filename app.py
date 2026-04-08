"""
Intelligent DSA Learning System - Flask Backend (Big Data Edition)
Main application with API routes for analysis, prediction, recommendations,
and Big Data pipeline status. Uses PySpark MLlib + MongoDB.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from leetcode_fetcher import fetch_all_user_data
from spark_ml import spark_predictor
from mongo_handler import mongo
from data_pipeline import pipeline
from recommender import recommend_questions, get_study_plan, analyze_topics
from ai_agent import ai_assistant

app = Flask(__name__)
CORS(app)

# ── Initialize Big Data components on startup ──
print("=" * 60)
print("  DSA Intelligence — Big Data Edition")
print("=" * 60)

# 1. Train Spark ML models
print("\n[1/3] Training ML models (PySpark MLlib)...")
spark_predictor.ensure_trained()

# 2. Store ML training metadata in MongoDB
ml_meta = spark_predictor.get_metadata()
mongo.store_ml_run(ml_meta)

# 3. Run initial data pipeline
print("\n[2/3] Running Spark batch data pipeline...")
pipeline_report = pipeline.run_batch_pipeline(100000)
mongo.store_pipeline_run(pipeline_report)

print("\n[3/3] All systems ready!")
print(f"  ML Engine: {ml_meta.get('engine', 'N/A')}")
print(f"  Dataset:   {ml_meta.get('dataset_size', 'N/A'):,} records")
print(f"  MongoDB:   {'Connected' if mongo.connected else 'In-Memory Fallback'}")
print("=" * 60)


@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_user():
    """
    Main analysis endpoint.
    Fetches LeetCode data, runs Spark ML prediction, stores in MongoDB.
    """
    data = request.get_json()
    username = data.get("username", "").strip()

    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Fetch LeetCode data via GraphQL
    user_data = fetch_all_user_data(username)
    if user_data is None:
        return jsonify({"error": f"User '{username}' not found on LeetCode"}), 404

    # ML Prediction (PySpark MLlib or scikit-learn fallback)
    prediction = spark_predictor.predict(user_data)

    # Topic Analysis
    topic_analysis = analyze_topics(user_data)

    # Study Plan
    study_plan = get_study_plan(user_data, prediction)

    # Store in MongoDB
    mongo.store_analysis(username, user_data, prediction, topic_analysis, study_plan)

    # Pipeline info for frontend
    pipeline_info = {
        "ml_metadata": spark_predictor.get_metadata(),
        "db_stats": mongo.get_db_stats(),
        "pipeline_summary": {
            "engine": pipeline.last_run.get("pipeline_engine", "N/A") if pipeline.last_run else "N/A",
            "records_processed": pipeline.last_run.get("records_processed", 0) if pipeline.last_run else 0,
            "processing_time": pipeline.last_run.get("processing_time_seconds", 0) if pipeline.last_run else 0
        }
    }

    return jsonify({
        "success": True,
        "user_data": user_data,
        "prediction": prediction,
        "topic_analysis": topic_analysis,
        "study_plan": study_plan,
        "pipeline_info": pipeline_info
    })


@app.route("/api/recommend", methods=["POST"])
def get_recommendations():
    """Get company-specific question recommendations."""
    data = request.get_json()
    username = data.get("username", "").strip()
    company = data.get("company", None)

    if not username:
        return jsonify({"error": "Username is required"}), 400

    user_data = fetch_all_user_data(username)
    if user_data is None:
        return jsonify({"error": f"User '{username}' not found on LeetCode"}), 404

    prediction = spark_predictor.predict(user_data)
    result = recommend_questions(user_data, target_company=company, prediction=prediction)

    return jsonify({
        "success": True,
        "recommendations": result["recommendations"],
        "topic_analysis": result["topic_analysis"],
        "target_difficulty": result["target_difficulty"]
    })


@app.route("/api/history/<username>", methods=["GET"])
def get_history(username):
    """Retrieve past analyses for a user from MongoDB."""
    history = mongo.get_user_history(username, limit=10)
    # Clean up for JSON serialization
    for record in history:
        record.pop("_id", None)
        record.pop("user_data", None)  # Too verbose for history view
        record.pop("topic_analysis", None)
    return jsonify({
        "success": True,
        "username": username,
        "history": history
    })


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    """Return Big Data pipeline status and statistics."""
    return jsonify({
        "ml_metadata": spark_predictor.get_metadata(),
        "db_stats": mongo.get_db_stats(),
        "last_pipeline_run": pipeline.get_last_run(),
        "pipeline_runs": pipeline.run_count
    })


@app.route("/api/companies", methods=["GET"])
def list_companies():
    """Return list of supported companies."""
    from recommender import load_company_questions
    companies = list(load_company_questions().keys())
    return jsonify({"companies": companies})

@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint for the AI Study Assistant."""
    data = request.get_json()
    message = data.get("message", "").strip()
    context_data = data.get("context_data", {})

    if not message:
        return jsonify({"error": "Message is required"}), 400

    reply = ai_assistant.get_response(message, context_data)
    
    return jsonify({
        "success": True,
        "reply": reply
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
