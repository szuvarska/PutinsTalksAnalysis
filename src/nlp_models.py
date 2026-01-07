from transformers import pipeline
import torch
import pycountry
import spacy

# A robust mapping for demonyms and common variations
DEMONYM_MAP = {
    "Russian": "Russia", "Soviets": "Russia", "Soviet": "Russia", "USSR": "Russia",
    "American": "USA", "Americans": "USA", "U.S.": "USA", "US": "USA", "United States": "USA",
    "Ukrainian": "Ukraine", "Ukrainians": "Ukraine",
    "Chinese": "China",
    "French": "France",
    "German": "Germany", "Germans": "Germany",
    "British": "UK", "Britons": "UK", "United Kingdom": "UK",
    "Polish": "Poland", "Poles": "Poland",
    "Belarusian": "Belarus",
    "Syrian": "Syria",
    "Turkish": "Turkey",
    "Iranian": "Iran",
    "Korean": "South Korea",  # Context dependent, but usually safe default in this dataset
    "North Korean": "North Korea",
    "Japanese": "Japan",
}

# Includes formal names ("Iran, Islamic Republic of"), common names ("Iran"), and aliases
VALID_COUNTRIES = set()
for c in pycountry.countries:
    VALID_COUNTRIES.add(c.name)
    if hasattr(c, 'common_name'):
        VALID_COUNTRIES.add(c.common_name)
    if ',' in c.name:  # e.g. "Iran, Islamic Republic of" -> "Iran"
        VALID_COUNTRIES.add(c.name.split(',')[0])

# Add missing common aliases explicitly
VALID_COUNTRIES.update(["Russia", "USA", "UK", "Syria", "Iran", "Turkey", "Vietnam", "South Korea", "North Korea"])

VALID_LANGUAGES = set([l.name for l in pycountry.languages])


def load_ner_pipeline(model_name="dslim/bert-base-NER", device=None):
    """Loads the BERT NER pipeline."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    print(f"Loading NER model: {model_name} on device {device}...")
    return pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)


def normalize_entity(text):
    return text.strip().title()


def extract_countries_bert(text, ner_pipeline):
    found_countries = []

    # Process in chunks to handle long text
    chunk_size = 512
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    try:
        results = ner_pipeline(chunks)
        if chunks and isinstance(results[0], dict):
            results = [results]

        # Allow ORG tags to catch things like "Russian Armed Forces"
        ALLOWED_TAGS = ['LOC', 'MISC', 'GPE', 'ORG']

        for chunk_res in results:
            for entity in chunk_res:
                if entity['entity_group'] in ALLOWED_TAGS:

                    raw_word = entity['word'].strip()
                    clean_word = raw_word.replace('.', '').replace(',', '')
                    norm_word = normalize_entity(clean_word)

                    # --- CHECK 1: Is it a Country? ---
                    if norm_word in VALID_COUNTRIES:
                        found_countries.append(norm_word)
                        continue

                    # --- CHECK 2: Is it in our Manual Map? ---
                    # Covers "Soviet", "British", "Americans"
                    if norm_word in DEMONYM_MAP:
                        found_countries.append(DEMONYM_MAP[norm_word])
                        continue

                    # --- CHECK 3: Is it a Language? ---
                    # If pycountry says it's a language (e.g., "German"), we treat it as valid.
                    # We might not know the country 100% (German -> Germany/Austria/Switzerland),
                    # so we default to looking it up in our DEMONYM_MAP.
                    if norm_word in VALID_LANGUAGES:
                        # If we have a specific mapping, use it (German -> Germany)
                        if norm_word in DEMONYM_MAP:
                            found_countries.append(DEMONYM_MAP[norm_word])

                    # --- CHECK 4: Split Multi-word Entities ---
                    # e.g., "Russian Armed Forces" -> checks "Russian"
                    sub_words = clean_word.split()
                    if len(sub_words) > 1:
                        for sub in sub_words:
                            norm_sub = normalize_entity(sub)
                            if norm_sub in DEMONYM_MAP:
                                found_countries.append(DEMONYM_MAP[norm_sub])
                            elif norm_sub in VALID_COUNTRIES:
                                found_countries.append(norm_sub)

    except Exception as e:
        print(f"Error: {e}")

    return found_countries


def load_zero_shot_pipeline(model_name="facebook/bart-large-mnli", device=None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=model_name, device=device)
