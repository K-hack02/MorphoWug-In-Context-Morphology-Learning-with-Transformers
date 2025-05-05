# In-Context Learning of Morphological Rules: A Computational Wug Test With Transformers

This project investigates how transformer models learn and apply morphological rules through in-context learning, inspired by the classic "Wug Test" in linguistics. We train a transformer model to recognize and apply various morphological transformations by learning patterns from example word pairs.

## Project Overview

Just as children can learn to apply linguistic rules to novel words (such as pluralizing the made-up word "wug" to "wugs"), transformer models can similarly learn morphological patterns through exposure to examples. This research explores the capabilities of neural networks in learning linguistic rules and provides insights into in-context learning in transformer models.

Our model is trained on numerous morphological transformation pairs and tested on its ability to generalize these rules to unseen words. For example, given a prompt like:
```
'dog is to dogs as cat is to cats as fox is to _____'
```
The model should correctly output 'foxes', demonstrating its understanding of plural formation rules.

## Morphological Transformations

This project includes a diverse set of morphological transformations commonly found in English and other languages:

1. **Singular to Plural**: Converting nouns from singular to plural form
   - Regular: dog → dogs, cat → cats, book → books
   - Irregular: mouse → mice, foot → feet, child → children
   - Special cases: fox → foxes, knife → knives, thesis → theses

2. **Present Tense to Past Tense**: Converting verbs from present to past tense
   - Regular: walk → walked, play → played, talk → talked
   - Irregular: go → went, see → saw, be → was
   - Special cases: try → tried, stop → stopped, lie → lay

3. **Adjective to Adverb**: Converting adjectives to their adverbial form
   - happy → happily, sad → sadly, clear → clearly
   - slow → slowly, quick → quickly, efficient → efficiently

4. **Verb to Noun**: Deriving nouns from verbs
   - teach → teacher, create → creation, swim → swimmer
   - act → action, investigate → investigation, manage → management

5. **Adjective to Noun**: Converting adjectives to related nouns
   - happy → happiness, brave → bravery, honest → honesty

6. **Base Case to Diminutive**: Forming diminutive forms
   - dog → doggy, cat → kitty, book → booklet

7. **Base to Past Participle**: Forming past participles from base forms
   - write → written, speak → spoken, fall → fallen

8. **Comparative to Superlative Adjective**: Converting comparative adjectives to superlative form
   - bigger → biggest, smaller → smallest, better → best

9. **Present Tense to Future Tense**: Converting verbs to future tense
   - walk → will walk, talk → will talk, eat → will eat

10. **Cardinal to Ordinal**: Converting numbers from cardinal to ordinal form
    - one → first, two → second, three → third

11. **Masculine to Feminine**: Converting masculine nouns to feminine forms
    - actor → actress, waiter → waitress, prince → princess

12. **Nominative to Accusative Pronoun**: Converting subject pronouns to object pronouns
    - I → me, he → him, she → her

13. **Positive to Negative Prefix**: Adding negative prefixes to words
    - happy → unhappy, certain → uncertain, fair → unfair

14. **Simple Past to Past Perfect**: Converting simple past to past perfect tense
    - walked → had walked, saw → had seen, ate → had eaten

15. **Affirmative to Negative**: Converting affirmative statements to negative form
    - is → is not, can → cannot, will → will not

And several other transformations that demonstrate the model's ability to learn various morphological rules.

## Running the Project in Google Colab

You can run this project directly in Google Colab: [Open in Colab](https://colab.research.google.com/github/adamoosya/182Proj/blob/main/run.ipynb)

### Running from Scratch

To run the model training from scratch:

1. Open the notebook in Google Colab
2. Run all cells in order
3. The model will be trained on the morphological transformation datasets
4. The training process includes:
   - Data preparation and loading
   - Model initialization
   - Training loop with loss tracking
   - Evaluation on test examples

This process may take several hours depending on the compute resources available in your Colab session.

### Loading from Checkpoint

To load a pre-trained model and skip the training process:

1. Open the notebook in Google Colab
2. Run the initial setup cells
3. Skip the "Train Model" and "Generate Loss/Accuracy" sections
4. Load the model from the checkpoint using:
   ```python
   # Load model from checkpoint
   model.load_state_dict(torch.load('checkpoint/epoch_5000.pth'))
   ```
5. Proceed with the evaluation cells to test the model's performance

Pre-trained model checkpoints are available at epochs 500, 1000, 1500, etc., up to epoch 5000.

## Project Structure

- `data/`: Contains CSV files for various morphological transformations
- `data.py`: Data handling and processing utilities
- `dataloader.py`: Data loading functions for model training
- `model.py`: Transformer model architecture
- `tokenizer.py`: Text tokenization functionality
- `hyperparameters.py`: Model hyperparameters
- `run.ipynb`: Main notebook for running experiments
- `checkpoint/`: Contains model checkpoints and training metrics

## Requirements

The project dependencies are listed in `requirements.txt`. When running in Colab, these will be automatically installed.

## Team Members

- Alec Thompson
- David Lee
- Kavin Vasudevan
- Adam Ousherovitch