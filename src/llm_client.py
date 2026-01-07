import os
import json
from google import genai
from google.genai import types
from env import GEMINI_API_KEY

class GeminiClient:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash"):
        """
        Initialize the Gemini Client using the new google-genai SDK.
        """
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            print("WARNING: No Google API Key provided. Set GOOGLE_API_KEY env var.")
            self.client = None
        else:
            # The new SDK uses a centralized Client object
            self.client = genai.Client(api_key=self.api_key)

        self.model_name = model_name

    def extract_countries(self, text):
        """
        Extracts ALL country mentions using Gemini.
        - Includes duplicates (frequency analysis).
        - Normalizes adjectives (Russian -> Russia).
        """
        if not self.client:
            return []

        prompt = f"""
        Analyze the following text from a political speech.

        Task:
        1. Identify EVERY mention of a country, nationality, or major geopolitical entity (like EU, NATO).
        2. Normalize them to the country/entity name (e.g., "Russian" -> "Russia", "Soviet" -> "Russia", "US" -> "USA").
        3. CRITICAL: Keep duplicates! If "Russia" is mentioned 5 times, your list must contain "Russia" 5 times.
        4. Return ONLY a valid JSON list of strings.

        Text:
        "{text[:10000]}" 
        """

        try:
            # New SDK Safety Settings structure
            # We map the old categories to the new 'types' structure if needed,
            # or use the simplified config strings if supported.
            # Here we use the types explicitly for clarity.

            # Note: The new SDK defaults are often sufficient, but here is how to permit content:
            safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
            ]

            # New API call structure: client.models.generate_content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    safety_settings=safety_settings
                )
            )

            # Parse JSON response
            # The new SDK response object has a .text property similar to the old one
            try:
                # Remove markdown fences if present
                clean_text = response.text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text.replace("```json", "").replace("```", "")
                elif clean_text.startswith("```"):
                    clean_text = clean_text.replace("```", "")

                countries = json.loads(clean_text)
            except (json.JSONDecodeError, AttributeError):
                print(f"Failed to parse JSON. Raw text: {response.text[:100]}...")
                return []

            if isinstance(countries, list):
                return countries
            elif isinstance(countries, dict) and 'countries' in countries:
                return countries['countries']
            return []

        except Exception as e:
            print(f"Gemini API Error: {e}")
            return []
