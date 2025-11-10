import google.generativeai as genai
from utils.get_env import get_google_api_key_env

# ✅ Configure Gemini API
genai.configure(api_key=get_google_api_key_env())

class ICON_FINDER_SERVICE:
    def __init__(self):
        # Using a lightweight, fast model (Gemini 1.5 Flash)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def suggest_icons(self, topic: str):
        """
        Suggests relevant emojis or icon keywords for the given topic.
        Works completely via Gemini — no vector DB needed.
        """
        prompt = f"""
        You are an expert presentation designer.
        Suggest 5 relevant emojis or short icon descriptions for the topic: "{topic}".
        Format your answer ONLY as a JSON array, e.g.:
        ["💡", "📊", "🤖", "🎓", "📈"]
        or
        ["light bulb", "robot head", "neural network", "graduation cap", "data chart"]
        """

        try:
            response = await self.model.generate_content_async(prompt)
            text = response.text.strip()

            # Parse the JSON safely
            if text.startswith("[") and text.endswith("]"):
                import json
                return json.loads(text)

            # Fallback default icons
            return ["💡", "📊", "🤖", "📈", "✨"]
        except Exception as e:
            print(f"⚠️ Icon generation failed: {e}")
            return ["💡", "📊", "🤖", "📈", "✨"]

ICON_FINDER_SERVICE = ICON_FINDER_SERVICE()

