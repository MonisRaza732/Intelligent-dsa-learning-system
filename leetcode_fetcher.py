"""
LeetCode GraphQL Data Fetcher
Fetches user profile, problem stats, and submission data from LeetCode's GraphQL API.
"""

import requests

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"

HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def fetch_user_profile(username):
    """Fetch basic user profile information."""
    query = """
    query getUserProfile($username: String!) {
        matchedUser(username: $username) {
            username
            profile {
                realName
                ranking
                userAvatar
                reputation
                starRating
            }
            submitStatsGlobal {
                acSubmissionNum {
                    difficulty
                    count
                }
            }
        }
    }
    """
    variables = {"username": username}
    try:
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if data.get("data", {}).get("matchedUser") is None:
            return None
        return data["data"]["matchedUser"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user profile: {e}")
        return None


def fetch_user_problem_stats(username):
    """Fetch problem-solving statistics broken down by difficulty."""
    query = """
    query userProblemsSolved($username: String!) {
        matchedUser(username: $username) {
            submitStatsGlobal {
                acSubmissionNum {
                    difficulty
                    count
                }
            }
        }
    }
    """
    variables = {"username": username}
    try:
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        user = data.get("data", {}).get("matchedUser")
        if user is None:
            return None
        stats = user["submitStatsGlobal"]["acSubmissionNum"]
        result = {}
        for item in stats:
            result[item["difficulty"]] = item["count"]
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error fetching problem stats: {e}")
        return None


def fetch_user_contest_info(username):
    """Fetch contest participation and rating information."""
    query = """
    query userContestRankingInfo($username: String!) {
        userContestRanking(username: $username) {
            attendedContestsCount
            rating
            globalRanking
            topPercentage
        }
        userContestRankingHistory(username: $username) {
            contest {
                title
                startTime
            }
            ranking
            rating
        }
    }
    """
    variables = {"username": username}
    try:
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        contest_ranking = data.get("data", {}).get("userContestRanking")
        contest_history = data.get("data", {}).get("userContestRankingHistory", [])
        return {
            "ranking": contest_ranking,
            "history": contest_history[-10:] if contest_history else []
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching contest info: {e}")
        return {"ranking": None, "history": []}


def fetch_skill_stats(username):
    """Fetch topic-wise skill tag stats."""
    query = """
    query skillStats($username: String!) {
        matchedUser(username: $username) {
            tagProblemCounts {
                advanced {
                    tagName
                    tagSlug
                    problemsSolved
                }
                intermediate {
                    tagName
                    tagSlug
                    problemsSolved
                }
                fundamental {
                    tagName
                    tagSlug
                    problemsSolved
                }
            }
        }
    }
    """
    variables = {"username": username}
    try:
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        user = data.get("data", {}).get("matchedUser")
        if user is None:
            return None
        return user.get("tagProblemCounts", {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching skill stats: {e}")
        return None


def fetch_recent_submissions(username, limit=20):
    """Fetch recent accepted submissions."""
    query = """
    query recentAcSubmissions($username: String!, $limit: Int!) {
        recentAcSubmissionList(username: $username, limit: $limit) {
            title
            titleSlug
            timestamp
            lang
        }
    }
    """
    variables = {"username": username, "limit": limit}
    try:
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("recentAcSubmissionList", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recent submissions: {e}")
        return []


def fetch_all_user_data(username):
    """Aggregate all user data into a single dictionary."""
    profile = fetch_user_profile(username)
    if profile is None:
        return None

    problem_stats = fetch_user_problem_stats(username)
    contest_info = fetch_user_contest_info(username)
    skill_stats = fetch_skill_stats(username)
    recent_submissions = fetch_recent_submissions(username)

    # Extract difficulty counts
    easy = problem_stats.get("Easy", 0) if problem_stats else 0
    medium = problem_stats.get("Medium", 0) if problem_stats else 0
    hard = problem_stats.get("Hard", 0) if problem_stats else 0
    total = problem_stats.get("All", 0) if problem_stats else 0

    # Extract contest info
    contest_rating = 0
    contests_attended = 0
    top_percentage = 100
    if contest_info and contest_info.get("ranking"):
        contest_rating = contest_info["ranking"].get("rating", 0)
        contests_attended = contest_info["ranking"].get("attendedContestsCount", 0)
        top_percentage = contest_info["ranking"].get("topPercentage", 100)

    # Process skill/tag data
    topics = {}
    if skill_stats:
        for level in ["fundamental", "intermediate", "advanced"]:
            for tag in skill_stats.get(level, []):
                topics[tag["tagName"]] = {
                    "solved": tag["problemsSolved"],
                    "level": level,
                    "slug": tag["tagSlug"]
                }

    ranking = profile.get("profile", {}).get("ranking", 0)
    if ranking is None or ranking == "N/A":
        ranking = 0

    return {
        "username": username,
        "profile": profile.get("profile", {}),
        "stats": {
            "easy": easy,
            "medium": medium,
            "hard": hard,
            "total": total
        },
        "contest": {
            "rating": round(contest_rating, 2) if contest_rating else 0,
            "attended": contests_attended,
            "top_percentage": round(top_percentage, 2) if top_percentage else 100
        },
        "topics": topics,
        "recent_submissions": recent_submissions,
        "ranking": ranking
    }
