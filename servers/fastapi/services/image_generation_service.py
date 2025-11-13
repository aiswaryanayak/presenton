import asyncio
import os
import aiohttp
import uuid
import google.generativeai as genai

from openai import AsyncOpenAI
from models.image_prompt import ImagePrompt
from models.sql.image_asset import ImageAsset
from utils.download_helpers import download_file
from utils.get_env import get_pexels_api_key_env, get_pixabay_api_key_env
from utils.image_provider import (
    is_pixels_selected,
    is_pixabay_selected,
    is_gemini_flash_selected,
    is_dalle3_selected,
)


# -----------------------------
# Google Config (important!)
# -----------------------------
# Must configure once, the API key is auto loaded from env
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print("⚠️ Failed to init Gemini:", e)


class ImageGenerationService:

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.image_gen_func = self.get_image_gen_func()

    def get_image_gen_func(self):
        """Decides which image provider to use."""
        if is_pixabay_selected():
            return self.get_image_from_pixabay
        elif is_pixels_selected():
            return self.get_image_from_pexels
        elif is_gemini_flash_selected():
            return self.generate_image_google  # <-- our fixed function
        elif is_dalle3_selected():
            return self.generate_image_openai
        return None

    def is_stock_provider_selected(self):
        return is_pixels_selected() or is_pixabay_selected()

    # =====================================================================
    # Main entry point → Generates image + handles storage & fallbacks
    # =====================================================================
    async def generate_image(self, prompt: ImagePrompt) -> str | ImageAsset:
        if not self.image_gen_func:
            print("⚠️ No image provider selected → using placeholder.")
            return "/static/images/placeholder.jpg"

        # AI providers get the themed prompt
        themed_prompt = prompt.get_image_prompt(
            with_theme=not self.is_stock_provider_selected()
        )
        print(f"🖼️ Image Request → {themed_prompt}")

        try:
            if self.is_stock_provider_selected():
                image_path = await self.image_gen_func(themed_prompt)
            else:
                image_path = await self.image_gen_func(themed_prompt, self.output_directory)

            if not image_path:
                return "/static/images/placeholder.jpg"

            # If it's a URL → return URL directly
            if isinstance(image_path, str) and image_path.startswith("http"):
                return image_path

            # If local file exists → wrap in ImageAsset
            if os.path.exists(image_path):
                return ImageAsset(
                    path=image_path,
                    is_uploaded=False,
                    extras={
                        "prompt": prompt.prompt,
                        "theme_prompt": prompt.theme_prompt,
                    },
                )

            return "/static/images/placeholder.jpg"

        except Exception as e:
            print("❌ Image generation error:", e)
            return "/static/images/placeholder.jpg"

    # =====================================================================
    # DALL-E 3
    # =====================================================================
    async def generate_image_openai(self, prompt: str, output_directory: str) -> str:
        client = AsyncOpenAI()
        result = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        image_url = result.data[0].url
        return await download_file(image_url, output_directory)

    # =====================================================================
    # GEMINI IMAGE GENERATION (FIXED)
    # =====================================================================
    async def generate_image_google(self, prompt: str, output_directory: str) -> str:
        """
        Uses the ONLY correct Gemini image model:
        gemini-2.0-flash-preview-image-generation
        """
        model_name = "gemini-2.0-flash-preview-image-generation"
        model = genai.GenerativeModel(model_name)

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"response_mime_type": "image/jpeg"},
        )

        image_path = None
        try:
            parts = response.candidates[0].content.parts
        except Exception:
            print("❌ Gemini image response malformed")
            return "/static/images/placeholder.jpg"

        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_path = os.path.join(output_directory, f"{uuid.uuid4()}.jpg")
                with open(image_path, "wb") as f:
                    f.write(part.inline_data.data)

        return image_path or "/static/images/placeholder.jpg"

    # =====================================================================
    # Pexels
    # =====================================================================
    async def get_image_from_pexels(self, prompt: str) -> str:
        async with aiohttp.ClientSession(trust_env=True) as session:
            response = await session.get(
                f"https://api.pexels.com/v1/search?query={prompt}&per_page=1",
                headers={"Authorization": get_pexels_api_key_env()},
            )
            data = await response.json()
            return data["photos"][0]["src"]["large"]

    # =====================================================================
    # Pixabay
    # =====================================================================
    async def get_image_from_pixabay(self, prompt: str) -> str:
        async with aiohttp.ClientSession(trust_env=True) as session:
            response = await session.get(
                f"https://pixabay.com/api/?key={get_pixabay_api_key_env()}&q={prompt}&image_type=photo&per_page=3"
            )
            data = await response.json()
            return data["hits"][0]["largeImageURL"]

