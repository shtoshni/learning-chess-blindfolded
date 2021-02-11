
# LENGTH_CATEGORIES = ["short", "long"]
LENGTH_CATEGORIES = ["long"]
TASK_CATEGORIES = ["end", "start"]
MOVE_TYPES = ["actual", "other"]

NOTATION_TO_REGEX = {
    "uci": r'([PQRKNB]|[a-h][1-8]|[bnqkr]|" ")'
}

PIECE_TYPES = ['P', 'N', 'R', 'B', 'Q', 'K']

PIECE_TO_VAL = {
    'p': 0,
    'P': 1,
    'n': 2,
    'N': 3,
    'b': 4,
    'B': 5,
    'r': 6,
    'R': 7,
    'q': 8,
    'Q': 9,
    'k': 10,
    'K': 11
}

