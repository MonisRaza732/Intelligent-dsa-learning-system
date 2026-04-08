"""
Company-Specific DSA Question Recommender
Recommends questions based on user's strengths, weaknesses, and target companies.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_company_questions():
    """Load company question database."""
    filepath = os.path.join(DATA_DIR, "company_questions.json")
    with open(filepath, "r") as f:
        return json.load(f)


def analyze_topics(user_data):
    """Analyze topic-wise strengths and weaknesses."""
    topics = user_data.get("topics", {})
    if not topics:
        return {"strengths": [], "weaknesses": [], "all_topics": []}

    topic_list = []
    for name, info in topics.items():
        solved = info.get("solved", 0)
        level = info.get("level", "fundamental")
        level_weight = {"fundamental": 1, "intermediate": 2, "advanced": 3}.get(level, 1)
        score = solved * level_weight
        topic_list.append({
            "name": name,
            "solved": solved,
            "level": level,
            "score": score
        })

    # Sort by score descending
    topic_list.sort(key=lambda x: x["score"], reverse=True)

    # Top 5 = strengths, Bottom 5 = weaknesses
    strengths = topic_list[:5] if len(topic_list) >= 5 else topic_list
    weaknesses = topic_list[-5:] if len(topic_list) >= 5 else []

    # Make sure weaknesses doesn't overlap strengths
    strength_names = {t["name"] for t in strengths}
    weaknesses = [t for t in weaknesses if t["name"] not in strength_names]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "all_topics": topic_list
    }


def recommend_questions(user_data, target_company=None, prediction=None):
    """
    Recommend questions based on user profile and target company.
    Prioritizes weak topics and adjusts difficulty based on skill level.
    """
    company_db = load_company_questions()
    topic_analysis = analyze_topics(user_data)
    
    skill_level = "Beginner"
    if prediction:
        skill_level = prediction.get("skill_level", "Beginner")

    # Determine appropriate difficulty
    if skill_level == "Beginner":
        preferred_difficulties = ["Easy", "Medium"]
    elif skill_level == "Intermediate":
        preferred_difficulties = ["Medium", "Easy"]
    elif skill_level == "Advanced":
        preferred_difficulties = ["Medium", "Hard"]
    else:
        preferred_difficulties = ["Hard", "Medium"]

    weak_topic_names = {t["name"] for t in topic_analysis.get("weaknesses", [])}
    strong_topic_names = {t["name"] for t in topic_analysis.get("strengths", [])}

    recommendations = {}

    companies_to_process = [target_company] if target_company and target_company in company_db else list(company_db.keys())

    for company in companies_to_process:
        company_data = company_db[company]
        questions = company_data["questions"]
        focus_topics = company_data["focus_topics"]

        scored_questions = []
        for q in questions:
            score = 0
            q_topic = q.get("topic", "")
            q_diff = q.get("difficulty", "Medium")

            # Boost questions on weak topics (need more practice)
            if q_topic in weak_topic_names:
                score += 30

            # Boost questions matching company focus areas
            if q_topic in focus_topics:
                score += 20

            # Boost questions matching preferred difficulty
            if q_diff in preferred_difficulties:
                score += 15

            # Slight penalty for topics already strong in
            if q_topic in strong_topic_names:
                score -= 5

            scored_questions.append({
                **q,
                "relevance_score": score,
                "url": f"https://leetcode.com/problems/{q['slug']}/"
            })

        # Sort by relevance
        scored_questions.sort(key=lambda x: x["relevance_score"], reverse=True)
        recommendations[company] = scored_questions[:8]

    return {
        "recommendations": recommendations,
        "topic_analysis": topic_analysis,
        "target_difficulty": preferred_difficulties
    }


def get_study_plan(user_data, prediction=None):
    """Generate a personalized study plan based on user's current level."""
    topic_analysis = analyze_topics(user_data)
    stats = user_data.get("stats", {})
    total = stats.get("total", 0)

    skill_level = "Beginner"
    readiness = 0
    if prediction:
        skill_level = prediction.get("skill_level", "Beginner")
        readiness = prediction.get("placement_readiness", 0)

    plan = {
        "current_level": skill_level,
        "readiness_score": readiness,
        "daily_target": 0,
        "weekly_plan": [],
        "focus_areas": [],
        "milestones": []
    }

    # Set daily targets based on level
    if skill_level == "Beginner":
        plan["daily_target"] = 2
        plan["weekly_plan"] = [
            "Mon-Tue: Arrays & Strings (Easy)",
            "Wed-Thu: Linked Lists & Stacks (Easy)",
            "Fri: Hash Maps & Sets (Easy-Medium)",
            "Sat: Practice contest problems",
            "Sun: Review & revise weak areas"
        ]
    elif skill_level == "Intermediate":
        plan["daily_target"] = 3
        plan["weekly_plan"] = [
            "Mon: Trees & Graphs (Medium)",
            "Tue: Dynamic Programming (Medium)",
            "Wed: Binary Search & Sorting (Medium)",
            "Thu: Backtracking & Recursion (Medium)",
            "Fri: Mixed difficulty contest prep",
            "Sat: Hard problem attempts",
            "Sun: Review solutions & study editorials"
        ]
    elif skill_level == "Advanced":
        plan["daily_target"] = 4
        plan["weekly_plan"] = [
            "Mon: Advanced DP & Graph algorithms (Hard)",
            "Tue: System Design + Trie/Segment Tree",
            "Wed: Competitive programming practice",
            "Thu: Company-specific question sets",
            "Fri: Mock interview practice",
            "Sat: Virtual contests",
            "Sun: Optimize solutions & learn new patterns"
        ]
    else:
        plan["daily_target"] = 3
        plan["weekly_plan"] = [
            "Mon-Tue: Maintain skills with Hard problems",
            "Wed: Explore advanced algorithms",
            "Thu: Mentor/help others (teaching solidifies learning)",
            "Fri: Virtual contests & upsolving",
            "Sat-Sun: System design & interview prep"
        ]

    # Focus areas from weaknesses
    weaknesses = topic_analysis.get("weaknesses", [])
    plan["focus_areas"] = [w["name"] for w in weaknesses[:5]]

    # Set milestones
    if total < 50:
        plan["milestones"] = [
            f"Solve 50 problems (currently {total})",
            "Complete all Easy array problems",
            "Attempt your first contest"
        ]
    elif total < 150:
        plan["milestones"] = [
            f"Solve 150 problems (currently {total})",
            "Solve 50+ Medium problems",
            "Achieve 1400+ contest rating"
        ]
    elif total < 300:
        plan["milestones"] = [
            f"Solve 300 problems (currently {total})",
            "Solve 20+ Hard problems",
            "Achieve 1700+ contest rating"
        ]
    else:
        plan["milestones"] = [
            "Maintain consistency with 3+ problems/day",
            "Achieve 2000+ contest rating",
            "Master all major DSA patterns"
        ]

    return plan
