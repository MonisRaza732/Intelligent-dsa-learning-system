import requests
import json
from collections import Counter

GITHUB_API_BASE = "https://api.github.com/users"
# Optional: Use a token for higher rate limits if provided via environment
import os
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

def get_github_headers():
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def fetch_github_data(username):
    """
    Fetches GitHub public profile data, sums up repository stars,
    and calculates the most used programming languages.
    """
    if not username:
        return None

    try:
        # 1. Fetch user profile
        profile_url = f"{GITHUB_API_BASE}/{username}"
        profile_res = requests.get(profile_url, headers=get_github_headers(), timeout=10)
        
        if profile_res.status_code != 200:
            print(f"[GITHUB] Failed to fetch profile for {username}: {profile_res.status_code}")
            return None
            
        profile_data = profile_res.json()
        
        # 2. Fetch user repositories (up to 100)
        repos_url = f"{GITHUB_API_BASE}/{username}/repos?per_page=100&sort=pushed"
        repos_res = requests.get(repos_url, headers=get_github_headers(), timeout=10)
        
        total_stars = 0
        total_forks = 0
        languages = Counter()
        
        if repos_res.status_code == 200:
            repos_data = repos_res.json()
            for repo in repos_data:
                # Disregard forks in the star count to measure original impact
                if not repo.get("fork", False):
                    total_stars += repo.get("stargazers_count", 0)
                    total_forks += repo.get("forks_count", 0)
                    
                    lang = repo.get("language")
                    if lang:
                        languages[lang] += 1
                        
        # Get top 3 languages
        top_languages = [lang for lang, count in languages.most_common(3)]
        
        return {
            "username": profile_data.get("login"),
            "avatar_url": profile_data.get("avatar_url"),
            "public_repos": profile_data.get("public_repos", 0),
            "followers": profile_data.get("followers", 0),
            "total_stars": total_stars,
            "total_forks": total_forks,
            "top_languages": top_languages,
            "github_url": profile_data.get("html_url")
        }
        
    except Exception as e:
        print(f"[GITHUB] Error fetching data for {username}: {e}")
        return None

if __name__ == "__main__":
    # Test block
    print(json.dumps(fetch_github_data("octocat"), indent=2))
