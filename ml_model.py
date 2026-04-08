"""
ML Model for DSA Skill Prediction
Uses scikit-learn to predict skill level and placement readiness from LeetCode stats.
Trains on synthetic data generated from realistic distributions.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def generate_synthetic_data(n_samples=2000):
    """Generate synthetic training data based on realistic LeetCode user distributions."""
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        # Random skill tier
        tier = np.random.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10])

        if tier == 0:  # Beginner
            easy = np.random.randint(0, 50)
            medium = np.random.randint(0, 15)
            hard = np.random.randint(0, 3)
            contest_rating = np.random.uniform(0, 1300)
            contests = np.random.randint(0, 5)
            topics_covered = np.random.randint(1, 6)
            consistency = np.random.uniform(0, 0.3)
        elif tier == 1:  # Intermediate
            easy = np.random.randint(30, 120)
            medium = np.random.randint(15, 80)
            hard = np.random.randint(2, 15)
            contest_rating = np.random.uniform(1200, 1600)
            contests = np.random.randint(3, 20)
            topics_covered = np.random.randint(5, 12)
            consistency = np.random.uniform(0.2, 0.6)
        elif tier == 2:  # Advanced
            easy = np.random.randint(80, 200)
            medium = np.random.randint(60, 200)
            hard = np.random.randint(10, 60)
            contest_rating = np.random.uniform(1500, 2000)
            contests = np.random.randint(10, 50)
            topics_covered = np.random.randint(10, 18)
            consistency = np.random.uniform(0.5, 0.85)
        else:  # Expert
            easy = np.random.randint(150, 300)
            medium = np.random.randint(150, 400)
            hard = np.random.randint(40, 150)
            contest_rating = np.random.uniform(1900, 2800)
            contests = np.random.randint(30, 100)
            topics_covered = np.random.randint(15, 22)
            consistency = np.random.uniform(0.7, 1.0)

        total = easy + medium + hard
        medium_ratio = medium / max(total, 1)
        hard_ratio = hard / max(total, 1)

        # Placement readiness score (0-100)
        readiness = min(100, (
            tier * 18 +
            min(total / 5, 20) +
            medium_ratio * 25 +
            hard_ratio * 20 +
            min(contest_rating / 100, 15) +
            consistency * 10 +
            np.random.normal(0, 3)
        ))
        readiness = max(0, readiness)

        data.append([
            easy, medium, hard, total,
            medium_ratio, hard_ratio,
            contest_rating, contests,
            topics_covered, consistency,
            tier, readiness
        ])

    return np.array(data)


class DSAPredictor:
    """ML model that predicts skill level and placement readiness."""

    SKILL_LABELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

    def __init__(self):
        self.skill_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42
        )
        self.readiness_regressor = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self):
        """Train both models on synthetic data."""
        data = generate_synthetic_data(2000)

        X = data[:, :10]  # Features
        y_skill = data[:, 10].astype(int)  # Skill tier
        y_readiness = data[:, 11]  # Readiness score

        X_scaled = self.scaler.fit_transform(X)

        # Train skill classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_skill, test_size=0.2, random_state=42
        )
        self.skill_classifier.fit(X_train, y_train)
        skill_accuracy = self.skill_classifier.score(X_test, y_test)

        # Train readiness regressor
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_scaled, y_readiness, test_size=0.2, random_state=42
        )
        self.readiness_regressor.fit(X_train2, y_train2)
        readiness_r2 = self.readiness_regressor.score(X_test2, y_test2)

        self.is_trained = True
        print(f"[ML] Skill Classifier Accuracy: {skill_accuracy:.3f}")
        print(f"[ML] Readiness Regressor R²: {readiness_r2:.3f}")

        # Save models
        self._save_models()
        return skill_accuracy, readiness_r2

    def _save_models(self):
        """Persist trained models to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.skill_classifier, os.path.join(MODEL_DIR, "skill_clf.pkl"))
        joblib.dump(self.readiness_regressor, os.path.join(MODEL_DIR, "readiness_reg.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    def _load_models(self):
        """Load persisted models from disk."""
        try:
            self.skill_classifier = joblib.load(os.path.join(MODEL_DIR, "skill_clf.pkl"))
            self.readiness_regressor = joblib.load(os.path.join(MODEL_DIR, "readiness_reg.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def ensure_trained(self):
        """Make sure models are ready to predict."""
        if not self.is_trained:
            if not self._load_models():
                self.train()

    def _extract_features(self, user_data):
        """Extract ML features from user data dict."""
        stats = user_data.get("stats", {})
        contest = user_data.get("contest", {})
        topics = user_data.get("topics", {})

        easy = stats.get("easy", 0)
        medium = stats.get("medium", 0)
        hard = stats.get("hard", 0)
        total = stats.get("total", 0) or (easy + medium + hard)
        medium_ratio = medium / max(total, 1)
        hard_ratio = hard / max(total, 1)
        contest_rating = contest.get("rating", 0) or 0
        contests_attended = contest.get("attended", 0) or 0
        topics_covered = len(topics)

        # Estimate consistency (0-1) from recent submissions
        recent = user_data.get("recent_submissions", [])
        consistency = min(len(recent) / 20, 1.0)

        return np.array([[
            easy, medium, hard, total,
            medium_ratio, hard_ratio,
            contest_rating, contests_attended,
            topics_covered, consistency
        ]])

    def predict(self, user_data):
        """Predict skill level and placement readiness for a user."""
        self.ensure_trained()

        features = self._extract_features(user_data)
        features_scaled = self.scaler.transform(features)

        skill_idx = self.skill_classifier.predict(features_scaled)[0]
        skill_proba = self.skill_classifier.predict_proba(features_scaled)[0]
        readiness = self.readiness_regressor.predict(features_scaled)[0]
        readiness = max(0, min(100, readiness))

        # Compute confidence levels for each skill tier
        confidence = {
            label: round(float(prob) * 100, 1)
            for label, prob in zip(self.SKILL_LABELS, skill_proba)
        }

        return {
            "skill_level": self.SKILL_LABELS[skill_idx],
            "skill_index": int(skill_idx),
            "confidence": confidence,
            "placement_readiness": round(float(readiness), 1),
            "readiness_label": self._readiness_label(readiness)
        }

    def _readiness_label(self, score):
        """Convert readiness score to a human-readable label."""
        if score >= 80:
            return "Highly Ready"
        elif score >= 60:
            return "Moderately Ready"
        elif score >= 40:
            return "Developing"
        elif score >= 20:
            return "Early Stage"
        else:
            return "Just Starting"


# Singleton instance
predictor = DSAPredictor()
