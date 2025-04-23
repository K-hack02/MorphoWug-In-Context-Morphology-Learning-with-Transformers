IS_TO_TOKEN = '#'
AS_TOKEN = '$'
END_TOKEN = '@'

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
    '#': 27,
    '$': 28,
    '@': 29,
    '\'': 30
}

TOKEN_TO_CHAR = {v: k for k, v in CHAR_TO_TOKEN.items()}

VOCAB_SIZE = len(CHAR_TO_TOKEN)

def tokenize(text):
    return [CHAR_TO_TOKEN[c] for c in text]

def detokenize(tokens):
    return ''.join([TOKEN_TO_CHAR[t] for t in tokens])

# if __name__ == "__main__":
#     print(tokenize("hello world"))
#     print(detokenize(tokenize("hello world")))
