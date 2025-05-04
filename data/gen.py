from google import genai
import os
from dotenv import load_dotenv
from schema import WordPairs, WordPair
import json
import time
from typing import Dict, List
import csv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DATA_PER_CATEGORY = 5_000
QUERY_BATCH_SIZE = 200
NUM_BATCHES = DATA_PER_CATEGORY // QUERY_BATCH_SIZE

data: Dict[str, List[str]] = {
    "singular_to_plural": set(),
    "present_tense_to_past_tense": set(),
    "base_case_to_third_person_singular": set(),
    "singular_possessive_to_plural_possessive": set(),
    "comparative_adjective_to_superlative_adjective": set(),
    "verb_to_progressive_verb": set(),
    "verb_to_derived_agentive": set(),
    "base_case_to_diminuitive": set(),
    "adjective_to_adverb": set(),
    "verb_to_gerund": set(),
    "noun_to_adjective": set(),
    "positive_to_negative_prefix": set(),
    "verb_to_noun": set(),
    "present_tense_to_future_tense": set(),
    "cardinal_to_ordinal": set(),
    "adjective_to_noun": set(),
    "base_to_reflexive_pronoun": set(),
    "nominative_to_accusative_pronoun": set(),
    "base_to_past_participle": set(),
    "simple_past_to_past_perfect": set(),
    "affirmative_to_negative": set(),
    "masculine_to_feminine": set(),
    "concrete_to_abstract_noun": set(),
}

def generate_word_pairs(category: str, num_pairs: int = QUERY_BATCH_SIZE) -> Dict[str, str]:
    """Generate word pairs for a given category."""
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=f'Generate {num_pairs} word pairs for the category "{category}". \
        You should return a list of word pairs, where each element is WordPair that contains two words (word1, word2). \
        Please be creative and varied in your word choices.',
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[WordPair],
            'temperature': 2.0,  # Set to maximum value for maximum randomness
        },
    )
    return response.parsed

def populate_data(categories: List[str] = list(data.keys()), num_batches: int = NUM_BATCHES, num_pairs: int = QUERY_BATCH_SIZE):
    for category in categories:
        for i in range(num_batches):
            print(f"Generating {num_pairs} word pairs for category: {category}, batch {i+1}/{num_batches}")
            pairs = generate_word_pairs(category, num_pairs)
            for pair in pairs:
                data[category].add((pair.word1, pair.word2))
            time.sleep(1)
            
        filename = os.path.join('data', f"{category}.csv")
        
        print(f"Saving {len(data[category])} pairs to {filename}")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for word1, word2 in data[category]:
                writer.writerow([word1, word2])
                
populate_data()
