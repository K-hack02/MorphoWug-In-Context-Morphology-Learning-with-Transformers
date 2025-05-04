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

DATA_PER_CATEGORY = 10_000
QUERY_BATCH_SIZE = 100
NUM_BATCHES = DATA_PER_CATEGORY // QUERY_BATCH_SIZE

data: Dict[str, List[str]] = {
    "singular to plural": [],
    "present tense to past tense": [],
    "base case to third person singular": [],
    "singular possessive to plural possessive": [],
    "comparative adjective to superlative adjective": [],
    "verb to progressive verb": [],
    "verb to derived agentive": [],
    "base case to diminuitive": [],
}

def generate_word_pairs(category: str, num_pairs: int = QUERY_BATCH_SIZE) -> Dict[str, str]:
    """Generate word pairs for a given category."""
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=f'Generate {num_pairs} word pairs for the category "{category}". \
        You should return a list of word pairs, where each element is WordPair that contains two words (word1, word2). \
        Please be creative and varied in your word choices.',
        generation_config={
            'temperature': 0.9,  # Higher temperature (0-1) means more random/creative outputs
            'candidate_count': 1,
        },
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[WordPair],
        },
    )
    return response.parsed

def populate_data(categories: List[str] = list(data.keys()), num_batches: int = NUM_BATCHES, num_pairs: int = QUERY_BATCH_SIZE):
    for category in categories:
        for i in range(num_batches):
            print(f"Generating {num_pairs} word pairs for category: {category}, batch {i+1}/{num_batches}")
            pairs = generate_word_pairs(category, num_pairs)
            data[category].extend(pairs)
            time.sleep(1)

    for category, pairs in data.items():
        # Create filename from category, replace spaces with underscores
        filename = os.path.join('data', f"{category.replace(' ', '_')}.csv")
        
        print(f"Saving {len(pairs)} pairs to {filename}")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['word1', 'word2'])  # Write header
            for pair in pairs:
                writer.writerow([pair.word1, pair.word2])
                
populate_data()
