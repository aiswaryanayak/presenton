import google.generativeai as genai


async def list_available_google_models(api_key: str) -> list[str]:
    """
    Lists available Gemini models using the Google Generative AI client.
    """
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()

        model_names = [m.name for m in models if "generateContent" in m.supported_generation_methods]
        return model_names or ["gemini-2.0-flash-exp"]

    except Exception as e:
        print(f"❌ Error fetching Gemini models: {str(e)}")
        # Fallback to a known working model
        return ["gemini-2.0-flash-exp"]
