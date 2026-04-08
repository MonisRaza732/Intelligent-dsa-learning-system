"""
Spark Batch Data Processing Pipeline
Performs large-scale data generation, transformation, and aggregation using PySpark.
Demonstrates Big Data processing patterns: distributed compute, Spark SQL, aggregations.
"""

import time
import os
import numpy as np

SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
except ImportError:
    pass


class DataPipeline:
    """
    Spark-based batch processing pipeline for DSA analytics.
    Generates, transforms, and aggregates large-scale user data.
    """

    def __init__(self):
        self.spark = None
        self.last_run = None
        self.run_count = 0

    def _get_spark(self):
        """Get or create SparkSession."""
        if self.spark is None:
            if not SPARK_AVAILABLE:
                return None
            try:
                self.spark = (SparkSession.builder
                              .appName("DSA-Pipeline")
                              .master("local[*]")
                              .config("spark.driver.memory", "2g")
                              .config("spark.ui.showConsoleProgress", "false")
                              .getOrCreate())
                self.spark.sparkContext.setLogLevel("ERROR")
            except Exception as e:
                print(f"[PIPELINE] Spark init failed: {e}")
                return None
        return self.spark

    def run_batch_pipeline(self, n_records=100000):
        """
        Execute the full batch processing pipeline:
        1. Generate large-scale data
        2. Create Spark DataFrame (distributed)
        3. Feature engineering with Spark SQL
        4. Aggregation & statistics
        5. Produce pipeline report
        """
        spark = self._get_spark()
        if spark is None:
            return self._run_fallback_pipeline(n_records)

        start_time = time.time()
        print(f"[PIPELINE] ═══ Starting Spark Batch Pipeline ({n_records:,} records) ═══")

        try:
            # ── Step 1: Generate raw data ──
            print("[PIPELINE] Step 1/5: Generating raw data...")
            np.random.seed(int(time.time()) % 10000)

            tiers = np.random.choice([0, 1, 2, 3], size=n_records, p=[0.30, 0.35, 0.25, 0.10])
            tier_names = {0: "Beginner", 1: "Intermediate", 2: "Advanced", 3: "Expert"}

            rows = []
            for i in range(n_records):
                tier = int(tiers[i])
                if tier == 0:
                    easy, medium, hard = np.random.randint(0, 50), np.random.randint(0, 15), np.random.randint(0, 3)
                    rating = float(np.random.uniform(0, 1300))
                elif tier == 1:
                    easy, medium, hard = np.random.randint(30, 120), np.random.randint(15, 80), np.random.randint(2, 15)
                    rating = float(np.random.uniform(1200, 1600))
                elif tier == 2:
                    easy, medium, hard = np.random.randint(80, 200), np.random.randint(60, 200), np.random.randint(10, 60)
                    rating = float(np.random.uniform(1500, 2000))
                else:
                    easy, medium, hard = np.random.randint(150, 300), np.random.randint(150, 400), np.random.randint(40, 150)
                    rating = float(np.random.uniform(1900, 2800))

                total = easy + medium + hard
                rows.append((int(easy), int(medium), int(hard), int(total), float(rating), tier, tier_names[tier]))

            # ── Step 2: Create Spark DataFrame ──
            print("[PIPELINE] Step 2/5: Creating distributed Spark DataFrame...")
            schema = StructType([
                StructField("easy", IntegerType(), False),
                StructField("medium", IntegerType(), False),
                StructField("hard", IntegerType(), False),
                StructField("total", IntegerType(), False),
                StructField("contest_rating", FloatType(), False),
                StructField("skill_tier", IntegerType(), False),
                StructField("tier_name", StringType(), False),
            ])

            df = spark.createDataFrame(rows, schema=schema)
            df = df.repartition(8)
            df.cache()

            actual_count = df.count()
            num_partitions = df.rdd.getNumPartitions()

            # ── Step 3: Feature Engineering via Spark SQL ──
            print("[PIPELINE] Step 3/5: Feature engineering (Spark SQL)...")
            df = df.withColumn("medium_ratio", F.col("medium") / F.greatest(F.col("total"), F.lit(1)))
            df = df.withColumn("hard_ratio", F.col("hard") / F.greatest(F.col("total"), F.lit(1)))
            df = df.withColumn("difficulty_score",
                               F.col("easy") * 1 + F.col("medium") * 3 + F.col("hard") * 5)
            df = df.withColumn("rating_bucket",
                               F.when(F.col("contest_rating") < 1300, "Bronze")
                                .when(F.col("contest_rating") < 1600, "Silver")
                                .when(F.col("contest_rating") < 2000, "Gold")
                                .otherwise("Platinum"))

            # ── Step 4: Aggregations ──
            print("[PIPELINE] Step 4/5: Running aggregations...")

            # Per-tier stats
            tier_stats = (df.groupBy("tier_name")
                         .agg(
                             F.count("*").alias("count"),
                             F.round(F.avg("total"), 1).alias("avg_total"),
                             F.round(F.avg("easy"), 1).alias("avg_easy"),
                             F.round(F.avg("medium"), 1).alias("avg_medium"),
                             F.round(F.avg("hard"), 1).alias("avg_hard"),
                             F.round(F.avg("contest_rating"), 1).alias("avg_rating"),
                             F.round(F.avg("difficulty_score"), 1).alias("avg_difficulty_score"),
                             F.max("total").alias("max_total"),
                             F.min("total").alias("min_total")
                         )
                         .orderBy("avg_total")
                         .collect())

            tier_data = [row.asDict() for row in tier_stats]

            # Rating bucket distribution
            rating_dist = (df.groupBy("rating_bucket")
                          .agg(F.count("*").alias("count"))
                          .orderBy("count", ascending=False)
                          .collect())

            rating_data = [row.asDict() for row in rating_dist]

            # Overall stats
            overall = df.agg(
                F.round(F.avg("total"), 1).alias("avg_total"),
                F.round(F.avg("contest_rating"), 1).alias("avg_rating"),
                F.round(F.stddev("total"), 1).alias("std_total"),
                F.max("total").alias("max_total"),
                F.min("total").alias("min_total")
            ).collect()[0].asDict()

            # ── Step 5: Report ──
            df.unpersist()
            elapsed = time.time() - start_time
            self.run_count += 1

            report = {
                "pipeline_engine": "Apache Spark",
                "spark_version": spark.version,
                "records_processed": actual_count,
                "num_partitions": num_partitions,
                "processing_time_seconds": round(elapsed, 2),
                "run_number": self.run_count,
                "tier_statistics": tier_data,
                "rating_distribution": rating_data,
                "overall_statistics": overall,
                "features_engineered": ["medium_ratio", "hard_ratio", "difficulty_score", "rating_bucket"],
                "transformations_applied": [
                    "Repartitioning (8 partitions)",
                    "Feature engineering (4 new columns)",
                    "Group-by aggregations (per-tier stats)",
                    "Rating bucket categorization",
                    "Statistical summaries"
                ]
            }

            self.last_run = report
            print(f"[PIPELINE] ═══ Pipeline complete: {actual_count:,} records in {elapsed:.1f}s ═══")
            return report

        except Exception as e:
            print(f"[PIPELINE] Error: {e}")
            return self._run_fallback_pipeline(n_records)

    def _run_fallback_pipeline(self, n_records):
        """Fallback pipeline using NumPy if Spark is unavailable."""
        start_time = time.time()
        n_records = min(n_records, 10000)  # Limit without Spark
        print(f"[PIPELINE] Running NumPy fallback pipeline ({n_records:,} records)...")

        np.random.seed(42)
        tiers = np.random.choice([0, 1, 2, 3], size=n_records, p=[0.30, 0.35, 0.25, 0.10])
        tier_names = {0: "Beginner", 1: "Intermediate", 2: "Advanced", 3: "Expert"}

        totals = []
        ratings = []
        tier_totals = {t: [] for t in tier_names.values()}

        for i in range(n_records):
            tier = tiers[i]
            if tier == 0:
                t = np.random.randint(0, 68)
                r = np.random.uniform(0, 1300)
            elif tier == 1:
                t = np.random.randint(47, 215)
                r = np.random.uniform(1200, 1600)
            elif tier == 2:
                t = np.random.randint(150, 460)
                r = np.random.uniform(1500, 2000)
            else:
                t = np.random.randint(340, 850)
                r = np.random.uniform(1900, 2800)
            totals.append(t)
            ratings.append(r)
            tier_totals[tier_names[tier]].append(t)

        elapsed = time.time() - start_time
        self.run_count += 1

        tier_data = []
        for name, vals in tier_totals.items():
            if vals:
                tier_data.append({
                    "tier_name": name,
                    "count": len(vals),
                    "avg_total": round(float(np.mean(vals)), 1),
                    "max_total": int(np.max(vals)),
                    "min_total": int(np.min(vals))
                })

        report = {
            "pipeline_engine": "NumPy (fallback)",
            "records_processed": n_records,
            "num_partitions": 1,
            "processing_time_seconds": round(elapsed, 2),
            "run_number": self.run_count,
            "tier_statistics": sorted(tier_data, key=lambda x: x["avg_total"]),
            "overall_statistics": {
                "avg_total": round(float(np.mean(totals)), 1),
                "avg_rating": round(float(np.mean(ratings)), 1),
                "max_total": int(np.max(totals)),
                "min_total": int(np.min(totals))
            },
            "transformations_applied": [
                "Data generation (NumPy)",
                "Statistical aggregations",
                "Per-tier breakdown"
            ]
        }

        self.last_run = report
        print(f"[PIPELINE] Fallback complete: {n_records:,} records in {elapsed:.2f}s")
        return report

    def get_last_run(self):
        """Return the last pipeline run report."""
        return self.last_run


# ── Singleton ──
pipeline = DataPipeline()
