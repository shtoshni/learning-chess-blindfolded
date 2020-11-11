import chess
import numpy as np


def detect_move_notation_file(source_file):
    """Detect move notation given the game file."""
    # Get the first 100 games or the minimum number of games
    games = [line.strip().split() for line in open(source_file).readlines()[:100]]
    all_moves = []
    for game in games:
        all_moves.extend(game)

    # print(all_moves)
    if '-' in all_moves[0]:
        # Must be a LAN game

        pawn_in_notation = False
        for move in all_moves:
            if move[0] == 'P':
                pawn_in_notation = True
                break

        if pawn_in_notation:
            return 'lan_with_p'
        else:
            return 'lan'
    else:
        try:
            board = chess.Board()
            move = board.parse_san(all_moves[0])
            move_san = board.san(move)
            if move_san == all_moves[0]:
                return 'san'
        except ValueError:
            pass

        # Must be a UCI variant
        piece_in_notation = False
        for move in all_moves:
            if move[0].isupper():
                piece_in_notation = True
                break

        if piece_in_notation:
            return 'rap'
        else:
            return 'uci'


def detect_move_notation_game(moves):
    """Detect move notation given the list of moves."""
    if '-' in moves[0]:
        # Must be a LAN game
        pawn_in_notation = False
        for move in moves:
            if move[0] == 'P':
                pawn_in_notation = True
                break
        if pawn_in_notation:
            return 'lan_with_p'
        else:
            return 'lan'
    else:
        try:
            board = chess.Board()
            move = board.parse_san(moves[0])
            move_san = board.san(move)
            if move_san == moves[0]:
                return 'san'
        except ValueError:
            pass

        # Must be a UCI variant
        piece_in_notation = False
        for move in moves:
            if move[0].isupper():
                piece_in_notation = True
                break

        if piece_in_notation:
            return 'rap'
        else:
            return 'uci'


def detect_move_notation(data):
    if isinstance(data, list):
        return detect_move_notation_game(data)
    elif isinstance(data, str):
        return detect_move_notation_file(data)
    else:
        raise ValueError(f"Data type {type(data)} not supported")

