import json
import asyncio
import google.generativeai as genai
from utils.get_env import get_google_api_key_env

# ✅ Configure Gemini API
genai.configure(api_key=get_google_api_key_env())


class IconFinderService:
    def __init__(self):
        # ⚡ Using Gemini 2.0 Experimental model
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    async def suggest_icons(self, topic: str) -> list[str]:
        """
        Suggest 5 relevant emojis or short icon keywords for the given topic.
        Uses Gemini 2.0 Flash Experimental.
        Returns a clean list of icons or keywords.
        """
        prompt = f"""
        You are an expert presentation designer.
        Suggest exactly 5 relevant emojis or short icon keywords for the topic: "{topic}".
        Format the response strictly as a JSON array, e.g.:
        ["💡", "📊", "🤖", "🎓", "📈"]
        """

        for attempt in range(2):  # Retry once if Gemini is slow or returns malformed JSON
            try:
                # ✅ New async API call format for Gemini 2.0 Exp
                response = await self.model.generate_content_async(
                    contents=prompt,
                    generation_config={"response_mime_type": "application/json"},
                )

                # Gemini 2.0 Exp sometimes returns structured JSON directly
                if hasattr(response, "candidates") and response.candidates:
                    text = response.candidates[0].content.parts[0].text
                else:
                    text = getattr(response, "text", "").strip()

                # Clean + parse JSON
                start, end = text.find("["), text.rfind("]")
                if start != -1 and end != -1:
                    icons_json = text[start:end + 1]
                    return json.loads(icons_json)

                return ["💡", "📊", "🤖", "📈", "✨"]

            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(1)
                else:
                    print(f"⚠️ Icon generation failed: {e}")
                    return ["💡", "📊", "🤖", "📈", "✨"]


# ✅ Global instance for reuse
icon_finder = IconFinderService()

# ✅ Backward-compatibility alias (required for process_slides.py)
ICON_FINDER_SERVICE = icon_finder

