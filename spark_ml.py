"""
PySpark MLlib — Big Data ML Pipeline for DSA Skill Prediction
Uses Apache Spark's MLlib for distributed model training on large-scale synthetic data.
Falls back to scikit-learn (ml_model.py) if PySpark is unavailable.
"""

import os
import time
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SPARK_MODEL_DIR = os.path.join(MODEL_DIR, "spark")

# ── Try importing PySpark ──
SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
    from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
    from pyspark.ml.classification import RandomForestClassifier as SparkRFC
    from pyspark.ml.regression import GBTRegressor as SparkGBTR
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
    SPARK_AVAILABLE = True
    print("[SPARK] PySpark is available — Big Data ML enabled")
except ImportError:
    print("[SPARK] PySpark not found — falling back to scikit-learn")


def get_spark_session():
    """Create or get a SparkSession configured for local execution."""
    return (SparkSession.builder
            .appName("DSA-Intelligence-BigData")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.log.level", "ERROR")
            .getOrCreate())


def generate_large_synthetic_data(n_samples=100000):
    """
    Generate a large-scale synthetic dataset (100K records).
    This is the Big Data version — 50x larger than the original 2K dataset.
    Returns a numpy array for conversion to Spark DataFrame.
    """
    np.random.seed(42)
    print(f"[SPARK] Generating {n_samples:,} synthetic training records...")

    tiers = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.30, 0.35, 0.25, 0.10])
    data = np.zeros((n_samples, 12))

    for i in range(n_samples):
        tier = tiers[i]
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

        readiness = min(100, max(0, (
            tier * 18 +
            min(total / 5, 20) +
            medium_ratio * 25 +
            hard_ratio * 20 +
            min(contest_rating / 100, 15) +
            consistency * 10 +
            np.random.normal(0, 3)
        )))

        data[i] = [
            easy, medium, hard, total,
            medium_ratio, hard_ratio,
            contest_rating, contests,
            topics_covered, consistency,
            tier, readiness
        ]

    return data


# ── Spark DataFrame Schema ──
SCHEMA = StructType([
    StructField("easy", FloatType(), False),
    StructField("medium", FloatType(), False),
    StructField("hard", FloatType(), False),
    StructField("total", FloatType(), False),
    StructField("medium_ratio", FloatType(), False),
    StructField("hard_ratio", FloatType(), False),
    StructField("contest_rating", FloatType(), False),
    StructField("contests", FloatType(), False),
    StructField("topics_covered", FloatType(), False),
    StructField("consistency", FloatType(), False),
    StructField("skill_tier", IntegerType(), False),
    StructField("readiness", FloatType(), False),
]) if SPARK_AVAILABLE else None

FEATURE_COLS = [
    "easy", "medium", "hard", "total",
    "medium_ratio", "hard_ratio",
    "contest_rating", "contests",
    "topics_covered", "consistency"
]


class SparkDSAPredictor:
    """
    Big Data ML predictor using PySpark MLlib.
    Trains RandomForestClassifier (skill) + GBTRegressor (readiness) on 100K records.
    Falls back to scikit-learn if Spark unavailable.
    """

    SKILL_LABELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

    def __init__(self):
        self.spark = None
        self.skill_pipeline = None
        self.readiness_pipeline = None
        self.is_trained = False
        self.training_metadata = {}
        self.use_spark = SPARK_AVAILABLE

        # Fallback scikit-learn models
        self._sklearn_predictor = None

    def _init_spark(self):
        """Initialize Spark session."""
        if self.spark is None and self.use_spark:
            try:
                self.spark = get_spark_session()
                self.spark.sparkContext.setLogLevel("ERROR")
            except Exception as e:
                print(f"[SPARK] Failed to initialize Spark: {e}")
                self.use_spark = False

    def train(self):
        """Train models — uses Spark if available, scikit-learn otherwise."""
        if self.use_spark:
            return self._train_spark()
        else:
            return self._train_sklearn_fallback()

    def _train_spark(self):
        """Train using PySpark MLlib on 100K records."""
        self._init_spark()
        if not self.use_spark:
            return self._train_sklearn_fallback()

        start_time = time.time()
        print("[SPARK] ═══ Starting Big Data ML Training Pipeline ═══")

        try:
            # Generate large dataset
            raw_data = generate_large_synthetic_data(100000)

            # Convert to Spark DataFrame
            print("[SPARK] Converting to Spark DataFrame...")
            rows = [tuple(float(v) if j != 10 else int(v) for j, v in enumerate(row))
                    for row in raw_data]
            df = self.spark.createDataFrame(rows, schema=SCHEMA)
            df.cache()

            record_count = df.count()
            print(f"[SPARK] Dataset size: {record_count:,} records")
            print(f"[SPARK] Partitions: {df.rdd.getNumPartitions()}")

            # ── Skill Classification Pipeline ──
            print("[SPARK] Training RandomForestClassifier (Skill Level)...")
            assembler_skill = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
            scaler_skill = SparkScaler(inputCol="raw_features", outputCol="features",
                                       withStd=True, withMean=True)
            rf = SparkRFC(
                featuresCol="features",
                labelCol="skill_tier",
                predictionCol="skill_prediction",
                probabilityCol="skill_probability",
                numTrees=100,
                maxDepth=12,
                seed=42
            )
            self.skill_pipeline = Pipeline(stages=[assembler_skill, scaler_skill, rf])

            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            skill_model = self.skill_pipeline.fit(train_df)
            self.skill_pipeline = skill_model

            # Evaluate skill model
            skill_preds = skill_model.transform(test_df)
            evaluator = MulticlassClassificationEvaluator(
                labelCol="skill_tier", predictionCol="skill_prediction",
                metricName="accuracy"
            )
            skill_accuracy = evaluator.evaluate(skill_preds)
            print(f"[SPARK] Skill Classifier Accuracy: {skill_accuracy:.4f}")

            # ── Readiness Regression Pipeline ──
            print("[SPARK] Training GBTRegressor (Placement Readiness)...")
            assembler_read = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features_r")
            scaler_read = SparkScaler(inputCol="raw_features_r", outputCol="features_r",
                                      withStd=True, withMean=True)
            gbt = SparkGBTR(
                featuresCol="features_r",
                labelCol="readiness",
                predictionCol="readiness_prediction",
                maxIter=100,
                maxDepth=6,
                seed=42
            )
            self.readiness_pipeline = Pipeline(stages=[assembler_read, scaler_read, gbt])

            readiness_model = self.readiness_pipeline.fit(train_df)
            self.readiness_pipeline = readiness_model

            # Evaluate readiness model
            readiness_preds = readiness_model.transform(test_df)
            reg_evaluator = RegressionEvaluator(
                labelCol="readiness", predictionCol="readiness_prediction",
                metricName="r2"
            )
            readiness_r2 = reg_evaluator.evaluate(readiness_preds)
            print(f"[SPARK] Readiness Regressor R²: {readiness_r2:.4f}")

            # Uncache
            df.unpersist()

            training_time = time.time() - start_time
            self.is_trained = True

            self.training_metadata = {
                "engine": "Apache Spark MLlib",
                "dataset_size": record_count,
                "features_used": len(FEATURE_COLS),
                "skill_accuracy": round(skill_accuracy, 4),
                "readiness_r2": round(readiness_r2, 4),
                "training_time_seconds": round(training_time, 2),
                "spark_version": self.spark.version,
                "num_partitions": df.rdd.getNumPartitions(),
                "models": {
                    "skill": "RandomForestClassifier (100 trees, depth=12)",
                    "readiness": "GBTRegressor (100 iterations, depth=6)"
                }
            }

            # Save Spark models
            self._save_models()

            print(f"[SPARK] ═══ Training complete in {training_time:.1f}s ═══")
            return skill_accuracy, readiness_r2

        except Exception as e:
            print(f"[SPARK] Training failed: {e}")
            print("[SPARK] Falling back to scikit-learn...")
            self.use_spark = False
            return self._train_sklearn_fallback()

    def _train_sklearn_fallback(self):
        """Fallback: train with scikit-learn if Spark is unavailable."""
        from ml_model import DSAPredictor
        print("[ML] Using scikit-learn fallback...")
        self._sklearn_predictor = DSAPredictor()
        result = self._sklearn_predictor.train()
        self.is_trained = True
        self.training_metadata = {
            "engine": "scikit-learn (fallback)",
            "dataset_size": 2000,
            "features_used": 10,
            "skill_accuracy": round(result[0], 4),
            "readiness_r2": round(result[1], 4),
            "training_time_seconds": 0,
            "models": {
                "skill": "RandomForestClassifier (scikit-learn)",
                "readiness": "GradientBoostingRegressor (scikit-learn)"
            }
        }
        return result

    def _save_models(self):
        """Save Spark pipeline models to disk."""
        os.makedirs(SPARK_MODEL_DIR, exist_ok=True)
        try:
            skill_path = os.path.join(SPARK_MODEL_DIR, "skill_pipeline")
            readiness_path = os.path.join(SPARK_MODEL_DIR, "readiness_pipeline")

            # Remove existing if any
            import shutil
            if os.path.exists(skill_path):
                shutil.rmtree(skill_path)
            if os.path.exists(readiness_path):
                shutil.rmtree(readiness_path)

            self.skill_pipeline.save(skill_path)
            self.readiness_pipeline.save(readiness_path)
            print("[SPARK] Models saved to disk")
        except Exception as e:
            print(f"[SPARK] Could not save models: {e}")

    def _load_models(self):
        """Attempt to load saved Spark models."""
        if not self.use_spark:
            return False
        try:
            self._init_spark()
            skill_path = os.path.join(SPARK_MODEL_DIR, "skill_pipeline")
            readiness_path = os.path.join(SPARK_MODEL_DIR, "readiness_pipeline")

            if os.path.exists(skill_path) and os.path.exists(readiness_path):
                self.skill_pipeline = PipelineModel.load(skill_path)
                self.readiness_pipeline = PipelineModel.load(readiness_path)
                self.is_trained = True
                print("[SPARK] Loaded saved models")
                return True
        except Exception as e:
            print(f"[SPARK] Could not load models: {e}")
        return False

    def ensure_trained(self):
        """Ensure models are ready for prediction."""
        if not self.is_trained:
            if not self._load_models():
                self.train()

    def predict(self, user_data):
        """Predict skill level and placement readiness."""
        self.ensure_trained()

        if self._sklearn_predictor:
            return self._sklearn_predictor.predict(user_data)

        if not self.use_spark:
            # Final fallback
            from ml_model import predictor as sklearn_predictor
            sklearn_predictor.ensure_trained()
            return sklearn_predictor.predict(user_data)

        return self._predict_spark(user_data)

    def _predict_spark(self, user_data):
        """Make prediction using Spark models."""
        stats = user_data.get("stats", {})
        contest = user_data.get("contest", {})
        topics = user_data.get("topics", {})
        recent = user_data.get("recent_submissions", [])

        easy = float(stats.get("easy", 0))
        medium = float(stats.get("medium", 0))
        hard = float(stats.get("hard", 0))
        total = float(stats.get("total", 0) or (easy + medium + hard))
        medium_ratio = medium / max(total, 1)
        hard_ratio = hard / max(total, 1)
        contest_rating = float(contest.get("rating", 0) or 0)
        contests_attended = float(contest.get("attended", 0) or 0)
        topics_covered = float(len(topics))
        consistency = min(len(recent) / 20.0, 1.0)

        # Create single-row DataFrame for prediction
        row_data = [(easy, medium, hard, total, medium_ratio, hard_ratio,
                     contest_rating, contests_attended, topics_covered, consistency,
                     0, 0.0)]  # dummy labels

        pred_df = self.spark.createDataFrame(row_data, schema=SCHEMA)

        # Skill prediction
        skill_result = self.skill_pipeline.transform(pred_df).collect()[0]
        skill_idx = int(skill_result["skill_prediction"])

        # Extract probability vector
        try:
            prob_vector = skill_result["skill_probability"]
            confidence = {
                label: round(float(prob_vector[i]) * 100, 1)
                for i, label in enumerate(self.SKILL_LABELS)
            }
        except Exception:
            confidence = {label: (100.0 if i == skill_idx else 0.0)
                         for i, label in enumerate(self.SKILL_LABELS)}

        # Readiness prediction
        readiness_result = self.readiness_pipeline.transform(pred_df).collect()[0]
        readiness = float(readiness_result["readiness_prediction"])
        readiness = max(0, min(100, readiness))

        return {
            "skill_level": self.SKILL_LABELS[skill_idx],
            "skill_index": skill_idx,
            "confidence": confidence,
            "placement_readiness": round(readiness, 1),
            "readiness_label": self._readiness_label(readiness),
            "ml_engine": "Apache Spark MLlib" if self.use_spark else "scikit-learn"
        }

    def _readiness_label(self, score):
        if score >= 80: return "Highly Ready"
        elif score >= 60: return "Moderately Ready"
        elif score >= 40: return "Developing"
        elif score >= 20: return "Early Stage"
        else: return "Just Starting"

    def get_metadata(self):
        """Return training metadata for the API."""
        return self.training_metadata


# ── Singleton ──
spark_predictor = SparkDSAPredictor()
