
GAME_TYPES = ["human", "synthetic"]
TRAIN_SIZE = {"small": 15_000, "medium": 100_000, "large": 200_000}
NOTATION_TO_REGEX = {
    "san": r'([ #=PRBNQKOx+-]|[a-h][1-8]|[a-h]|[1-8]||[bnqkr])',
    "lan": r'([ #=PRBNQKOx+-])',
    "lan_with_p": r'([ #=PRBNQKOx+-])',
    "uci": r'([a-h][1-8]|[bnqkr]|" ")',
    "rap": r'([PQRKNB]|[a-h][1-8]|[bnqkr]|" ")'
}

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
