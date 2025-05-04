from google import genai
import os
from dotenv import load_dotenv
from schema import WordPair
import time
from typing import Dict, List
import csv
import os
from hyperparameters import DATA_PER_CATEGORY, QUERY_BATCH_SIZE, NUM_BATCHES, DATA_CATEGORIES

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

data: Dict[str, List[str]] = {category: set() for category in DATA_CATEGORIES}

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
