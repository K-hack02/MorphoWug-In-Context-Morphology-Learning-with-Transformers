import torch

BATCH_SIZE = 64 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 128 # what is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'