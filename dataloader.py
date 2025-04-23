import torch
import numpy as np
from hyperparameters import BATCH_SIZE, BLOCK_SIZE, DEVICE
from tokenizer import tokenize, IS_TO_TOKEN, AS_TOKEN, END_TOKEN

def get_context_test(dataset, category=None):
    '''
    For example:
    context = fox$foxes#dog$
    target = dogs
    '''
    if category is None:
        category = np.random.choice(list(dataset.keys()))
    data = dataset[category]
    n = len(data)
    word1, word2 = data[np.random.randint(n)]
    word3, word4 = data[np.random.randint(n)]
    context = word1 + IS_TO_TOKEN + word2 + AS_TOKEN + word3 + IS_TO_TOKEN
    target = word4
    return context, target

def get_context_example(dataset, category=None):
    '''
    Data is of the form:
    singular $ plural # singular $ part of a plural

    For example:
    x = fox$foxes#dog$do
    y = ox$foxes#dog$dog
    '''
    if category is None:
        category = np.random.choice(list(dataset.keys()))
    data = dataset[category]
    n = len(data)
    word1, word2 = data[np.random.randint(n)]
    word3, word4 = data[np.random.randint(n)]
    window = np.random.randint(len(word4)+1)
    word4 += END_TOKEN
    x = word1 + IS_TO_TOKEN + word2 + AS_TOKEN + word3 + IS_TO_TOKEN + word4[:window]
    y = word1[1:] + IS_TO_TOKEN + word2 + AS_TOKEN + word3 + IS_TO_TOKEN + word4[:window + 1]
    x = torch.tensor(tokenize(x))
    y = torch.tensor(tokenize(y))
    return x, y

def get_batch(dataset, category=None, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=DEVICE):
    x = torch.zeros((batch_size, block_size), dtype=int)
    y = torch.zeros((batch_size, block_size), dtype=int)
    mask = torch.zeros((batch_size, block_size), dtype=torch.bool)
    for i in range(batch_size):
        xi, yi = get_context_example(dataset, category=category)
        x[i, :len(xi)] = xi  # Assign values up to the original length
        y[i, :len(yi)] = yi  # The rest will be padded with 0s
        mask[i, :len(xi)] = True  # Set True for actual data tokens

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask