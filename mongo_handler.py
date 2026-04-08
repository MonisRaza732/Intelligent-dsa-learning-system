"""
MongoDB Data Handler — NoSQL Persistence Layer
Stores user analysis results, ML training runs, and enables historical tracking.
Falls back to in-memory storage if MongoDB is unavailable.
"""

import os
import time
from datetime import datetime, timezone

# ── Try importing pymongo ──
MONGO_AVAILABLE = False
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    print("[MONGO] pymongo not installed — using in-memory storage")


class MongoHandler:
    """
    MongoDB data layer for persisting user analyses and ML metadata.
    Connects to a local MongoDB instance.  Falls back to an in-memory
    dict store if MongoDB is unavailable so the app always works.
    """

    def __init__(self, uri="mongodb://localhost:27017", db_name="dsa_intelligence"):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False

        # In-memory fallback
        self._memory_store = {
            "user_analyses": [],
            "ml_runs": [],
            "pipeline_runs": []
        }

        self._connect()

    def _connect(self):
        """Attempt to connect to MongoDB."""
        if not MONGO_AVAILABLE:
            print("[MONGO] Using in-memory fallback storage")
            return

        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]

            # Create indexes
            self.db.user_analyses.create_index("username")
            self.db.user_analyses.create_index("timestamp")
            self.db.ml_runs.create_index("timestamp")

            self.connected = True
            print(f"[MONGO] Connected to MongoDB at {self.uri}")
            print(f"[MONGO] Database: {self.db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            print(f"[MONGO] Could not connect to MongoDB: {e}")
            print("[MONGO] Using in-memory fallback storage")
            self.connected = False

    def store_analysis(self, username, user_data, prediction, topic_analysis, study_plan):
        """Store a user analysis result."""
        doc = {
            "username": username,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "user_data": user_data,
            "prediction": prediction,
            "topic_analysis": {
                "strengths": topic_analysis.get("strengths", []),
                "weaknesses": topic_analysis.get("weaknesses", []),
                "topic_count": len(topic_analysis.get("all_topics", []))
            },
            "study_plan_level": study_plan.get("current_level", ""),
            "stats_snapshot": {
                "total": user_data.get("stats", {}).get("total", 0),
                "easy": user_data.get("stats", {}).get("easy", 0),
                "medium": user_data.get("stats", {}).get("medium", 0),
                "hard": user_data.get("stats", {}).get("hard", 0),
                "contest_rating": user_data.get("contest", {}).get("rating", 0)
            }
        }

        if self.connected:
            try:
                result = self.db.user_analyses.insert_one(doc)
                doc["_id"] = str(result.inserted_id)
                print(f"[MONGO] Stored analysis for '{username}' (id: {doc['_id']})")
                return doc["_id"]
            except Exception as e:
                print(f"[MONGO] Store failed: {e}")

        # Fallback: in-memory
        self._memory_store["user_analyses"].append(doc)
        print(f"[MEMORY] Stored analysis for '{username}' (in-memory)")
        return f"mem_{len(self._memory_store['user_analyses'])}"

    def get_user_history(self, username, limit=10):
        """Retrieve past analyses for a user."""
        if self.connected:
            try:
                cursor = (self.db.user_analyses
                         .find({"username": username}, {"_id": 0})
                         .sort("epoch", -1)
                         .limit(limit))
                results = list(cursor)
                return results
            except Exception as e:
                print(f"[MONGO] Query failed: {e}")

        # Fallback
        user_records = [
            r for r in self._memory_store["user_analyses"]
            if r.get("username") == username
        ]
        user_records.sort(key=lambda x: x.get("epoch", 0), reverse=True)
        return user_records[:limit]

    def store_ml_run(self, metadata):
        """Store ML training run metadata."""
        doc = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            **metadata
        }

        if self.connected:
            try:
                self.db.ml_runs.insert_one(doc)
                print("[MONGO] Stored ML training run metadata")
                return True
            except Exception as e:
                print(f"[MONGO] ML run store failed: {e}")

        self._memory_store["ml_runs"].append(doc)
        return True

    def store_pipeline_run(self, pipeline_info):
        """Store a data pipeline run report."""
        doc = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            **pipeline_info
        }

        if self.connected:
            try:
                self.db.pipeline_runs.insert_one(doc)
                print("[MONGO] Stored pipeline run report")
                return True
            except Exception as e:
                print(f"[MONGO] Pipeline store failed: {e}")

        self._memory_store["pipeline_runs"].append(doc)
        return True

    def get_total_analyses_count(self):
        """Get total number of stored analyses."""
        if self.connected:
            try:
                return self.db.user_analyses.count_documents({})
            except Exception:
                pass
        return len(self._memory_store["user_analyses"])

    def get_unique_users_count(self):
        """Get count of unique users analyzed."""
        if self.connected:
            try:
                return len(self.db.user_analyses.distinct("username"))
            except Exception:
                pass
        usernames = set(r.get("username") for r in self._memory_store["user_analyses"])
        return len(usernames)

    def get_db_stats(self):
        """Get database statistics for the dashboard."""
        return {
            "connected": self.connected,
            "engine": "MongoDB" if self.connected else "In-Memory",
            "database": self.db_name if self.connected else "N/A",
            "total_analyses": self.get_total_analyses_count(),
            "unique_users": self.get_unique_users_count()
        }


# ── Singleton ──
mongo = MongoHandler()
