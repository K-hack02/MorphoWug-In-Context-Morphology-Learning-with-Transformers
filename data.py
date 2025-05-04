import csv
import os
import subprocess
from hyperparameters import DATA_CATEGORIES, TRAIN_DATA_CATEGORIES, VAL_DATA_CATEGORIES
from tokenizer import CHAR_TO_TOKEN

def load_data():
    data = {category: [] for category in DATA_CATEGORIES}
    train_data = {category: [] for category in TRAIN_DATA_CATEGORIES}
    val_data = {category: [] for category in VAL_DATA_CATEGORIES}
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download data files
    csv_filepaths = [category + ".csv" for category in DATA_CATEGORIES]
    for csv_filepath in csv_filepaths:
        url = f"https://raw.githubusercontent.com/adamoosya/182Proj/main/data/{csv_filepath}"
        output_path = os.path.join(data_dir, csv_filepath)
        try:
            # Use subprocess for downloading files
            subprocess.run(["wget", "--no-cache", "--backups=1", "-O", output_path, url], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"Error downloading {url}. Make sure wget is installed or modify the code to use another download method.")
            continue

    # Process the downloaded CSV files
    for category in DATA_CATEGORIES:
        csv_filepath = os.path.join(data_dir, category + ".csv")
        if not os.path.exists(csv_filepath):
            print(f"Warning: {csv_filepath} not found. Skipping.")
            continue
            
        with open(csv_filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            word_pairs = set()
            for row in reader:
                if len(row) >= 2:  # Ensure there are at least two elements in the row
                    word1, word2 = row[0].lower(), row[1].lower()
                    if all(c in CHAR_TO_TOKEN for c in word1) and all(c in CHAR_TO_TOKEN for c in word2):
                        word_pairs.add((word1, word2))
            word_pairs = list(word_pairs)
            data[category] = word_pairs

    for category, word_pairs in data.items():
        if category in TRAIN_DATA_CATEGORIES:
            split_index = int(0.9 * len(word_pairs))
            train_data[category] = word_pairs[:split_index]
            val_data[category] = word_pairs[split_index:]
        else:
            val_data[category] = word_pairs

    return data, train_data, val_data