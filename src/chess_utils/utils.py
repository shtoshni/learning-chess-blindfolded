import chess
from chess_utils.conversion_utils import convert_move_to_notation
import re
import sys


def get_move_prefix_size(move, notation='lan'):
    if notation == 'lan':
        return 3
    elif notation == 'uci':
        return 2
    elif notation == 'san':
        pattern_to_prefix = {r'^[NKQBR]x?[a-h][1-8].*': 1,
                             r'^[NKQBR]([a-h]|[1-8])x?[a-h][1-8].*': 1,
                             }
        pattern_to_prefix = {re.compile(pattern): prefix for pattern, prefix in pattern_to_prefix.items()}
        for pattern, prefix in pattern_to_prefix.items():
            if re.match(pattern, move):
                return prefix

        # NO MATCH!
        print(f"No match {move}")
        sys.exit()


def get_syntax_moves(board, location, notation="uci"):
    # print(location)
    # print(board)
    square_idx = chess.SQUARE_NAMES.index(location)
    piece_type = str(board.piece_at(square_idx)).upper()
    piece = chess.Piece.from_symbol(piece_type)

    # Clear the board
    board.clear()
    # Place the piece at its original square
    board.set_piece_at(square_idx, piece)

    legal_moves = list(board.pseudo_legal_moves)
    move_options = []
    for move in legal_moves:
        move_repr = convert_move_to_notation(move, board, notation=notation)
        move_prefix_size = get_move_prefix_size(move_repr, notation=notation)

        move_prefix, move_suffix = move_repr[:move_prefix_size], move_repr[move_prefix_size:]
        move_options.append(move_suffix)

    return move_options


def get_syntax_locations(board, piece_symbol):
    locations = []
    for square_name in chess.SQUARE_NAMES:
        square_idx = chess.SQUARE_NAMES.index(square_name)
        cur_piece_type = board.piece_at(square_idx)
        if cur_piece_type is None:
            continue

        cur_piece_type = str(cur_piece_type)
        if cur_piece_type == piece_symbol:
            locations.append(square_name)

    return locations
