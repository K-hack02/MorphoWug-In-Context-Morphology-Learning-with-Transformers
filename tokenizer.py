"""
Our data consists of spaces and lowercase letters. We want to tokenize the data into integers.
We also have two custom tokens: -> and ; to indicate the analogy format.
"""

CHAR_TO_TOKEN = {
    ' ': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'q': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'w': 23,
    'x': 24,
    'y': 25,
    'z': 26,
    '->': 27,
    ';': 28
}

TOKEN_TO_CHAR = {v: k for k, v in CHAR_TO_TOKEN.items()}

def tokenize(text):
    return [CHAR_TO_TOKEN[c] for c in text]

def detokenize(tokens):
    return ''.join([TOKEN_TO_CHAR[t] for t in tokens])

# if __name__ == "__main__":
#     print(tokenize("hello world"))
#     print(detokenize(tokenize("hello world")))
