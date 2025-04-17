# In-Context Learning of Morphological Rules: A Computational Wug Test With Transformers

## Team Members
- Alec Thompson
- David Lee
- Kavin Vasudevan
- Adam Ousherovitch

## Project Description
This project investigates the ability of transformer models to learn and apply morphological rules through in-context learning. Inspired by the classic "Wug Test" in linguistics, we train a transformer model on word pairs that follow specific morphological patterns (such as singular-to-plural forms and present-to-past tense verbs).

The model is tested on its ability to generalize these rules to novel words through in-context learning. For example, given a prompt like:
```
'dog is to dogs as cat is to cats as fox is to _____'
```
The model should correctly output 'foxes', demonstrating its understanding of the pluralization rule.

This research contributes to our understanding of how neural networks can learn and apply linguistic rules, and provides insights into the nature of in-context learning in transformer models.

## Quickstart Guide

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

5. Run the experiment:
```bash
jupyter lab Expirement.ipynb
```

## Project Structure
- `data/`: Contains dataset generation and processing scripts
- `Expirement.ipynb`: Main experiment notebook
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API keys, etc.)

## Requirements
- Python 3.x
- Jupyter Lab
- Dependencies listed in requirements.txt


