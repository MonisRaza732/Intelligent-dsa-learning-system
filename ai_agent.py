"""
AI Study Assistant — Backend Integration
Connects to Google Gemini (or uses an intelligent fallback mock) to provide
personalized DSA study advice using the user's specific context.
"""

import os
import time

# Optional real API integration (e.g., Google Gemini)
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    pass

class AIAssistant:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.use_real_ai = GENAI_AVAILABLE and bool(self.api_key)
        self.model = None

        if self.use_real_ai:
            try:
                genai.configure(api_key=self.api_key)
                # Use a fast text model
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("[AI] Initialized real Google Gemini AI Assistant.")
            except Exception as e:
                print(f"[AI] Failed to init Gemini: {e}")
                self.use_real_ai = False

        if not self.use_real_ai:
            print("[AI] Using Intelligent Fallback Mock (No API key found).")

    def get_response(self, user_message, context_data):
        """Get an AI response, injecting the user's LeetCode context."""
        
        # Build context string
        context_str = self._build_context(context_data)
        
        if self.use_real_ai:
            return self._get_gemini_response(user_message, context_str)
        else:
            return self._get_mock_response(user_message, context_str)

    def _build_context(self, context_data):
        """Convert raw user data into a prompt context."""
        if not context_data:
            return "No user context available."

        stats = context_data.get("stats", {})
        prediction = context_data.get("prediction", {})
        topics = context_data.get("topic_analysis", {})

        weaknesses = [t["name"] for t in topics.get("weaknesses", [])[:3]]
        strengths = [t["name"] for t in topics.get("strengths", [])[:3]]

        return f"""
Current User Context:
- Skill Tier: {prediction.get("skill_level", "Unknown")}
- Total Solved: {stats.get("total", 0)} (Easy: {stats.get("easy", 0)}, Medium: {stats.get("medium", 0)}, Hard: {stats.get("hard", 0)})
- Placement Readiness Score: {prediction.get("placement_readiness", 0)}/100
- Weak Topics: {", ".join(weaknesses) if weaknesses else "None yet"}
- Strong Topics: {", ".join(strengths) if strengths else "None yet"}
"""

    def _get_gemini_response(self, message, context):
        try:
            prompt = f"""
You are a friendly, expert DSA and Interview Prep AI Assistant integrated into the 'DSA Intelligence' web app.
Keep your answers relatively concise, encouraging, and formatted with markdown.

{context}

User says: {message}

AI Response:
"""
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[AI] Gemini generation failed: {e}")
            return "I'm having trouble connecting to my brain right now. Can we try again later?"

    def _get_mock_response(self, message, context):
        """Intelligent mock response if no API key is provided."""
        time.sleep(1.5) # Simulate network delay
        msg_lower = message.lower()
        
        # Parse context to give smart-sounding answers
        skill = "Beginner"
        if "Skill Tier: Intermediate" in context: skill = "Intermediate"
        elif "Skill Tier: Advanced" in context: skill = "Advanced"
        elif "Skill Tier: Expert" in context: skill = "Expert"

        if "hello" in msg_lower or "hi" in msg_lower:
            return f"Hello! 👋 I'm your AI Study Assistant. I see you're currently at an **{skill}** level. How can I help you level up today?"
        
        if "weak" in msg_lower or "improve" in msg_lower:
            if "Weak Topics:" in context:
                # Extract weak topics string hackily
                try:
                    weak = context.split("Weak Topics: ")[1].split("\n")[0]
                    return f"Based on your latest analysis, your weakest areas are **{weak}**. I recommend doing 2-3 medium difficulty questions in those topics this week!"
                except:
                    pass
            return "To improve, focus on patterns rather than memorizing solutions. Need help with a specific data structure?"

        if "interview" in msg_lower or "placement" in msg_lower or "job" in msg_lower:
             return f"You're making great progress! For interviews, make sure you can confidently explain the time/space complexity of every solution you write. Would you like me to quiz you on a specific pattern?"

        return f"That's a great question! *(Note: This is a fallback mock because no Gemini API key is configured. To get real AI answers, set `GEMINI_API_KEY` in your environment!)*"

# Singleton
ai_assistant = AIAssistant()
