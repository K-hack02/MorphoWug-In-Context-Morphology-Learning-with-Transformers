from google import genai
import os
from dotenv import load_dotenv
from schema import WordPairs
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='There are 2 different categories of word pairs. The first category is "singular to plural" \
    and the second category is "present tense to past tense". For each category, generate 10 word pairs.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[WordPairs],
    },
)

# Save the response to a JSON file
with open('data/dataset.json', 'w') as f:
    json.dump(response.text, f, indent=2)