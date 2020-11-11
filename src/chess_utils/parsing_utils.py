import chess
from chess_utils.conversion_utils import convert_lan_to_uci


def return_lan_valid_prefix(game_lan):
    if isinstance(game_lan, str):
        game_lan = game_lan.strip().split()

    uci_game = convert_lan_to_uci(game_lan)
    board = chess.Board()
    move_counter = 0
    try:
        for move in uci_game:
            board.push_uci(move)
            move_counter += 1
    except ValueError as e:
        pass

    return game_lan[:move_counter], move_counter


def return_san_valid_prefix(game_san):
    if isinstance(game_san, str):
        game_san = game_san.strip().split()

    board = chess.Board()
    move_counter = 0
    try:
        for move in game_san:
            board.push_san(move)
            move_counter += 1
    except ValueError as e:
        pass

    return game_san[:move_counter], move_counter


def return_uci_valid_prefix(game_uci):
    if isinstance(game_uci, str):
        game_uci = game_uci.strip().split()

    board = chess.Board()
    move_counter = 0
    try:
        for move in game_uci:
            board.push_uci(move)
            move_counter += 1
    except ValueError as e:
        pass

    return game_uci[:move_counter], move_counter


def return_rap_valid_prefix(game_rap):
    if isinstance(game_rap, str):
        game_rap = game_rap.strip().split()

    board = chess.Board()
    move_counter = 0
    try:
        for move in game_rap:
            # Remove piece name if it is part of the notation
            move = move[1:] if move[0].isupper() else move
            board.push_uci(move)
            move_counter += 1
    except ValueError as e:
        pass

    return game_rap[:move_counter], move_counter


def return_valid_prefix(game, notation='lan'):
    if notation == 'lan':
        return return_lan_valid_prefix(game)
    elif notation == 'san':
        return return_san_valid_prefix(game)
    elif notation == 'uci':
        return return_uci_valid_prefix(game)
    elif notation == 'rap':
        return return_rap_valid_prefix(game)
    else:
        raise ValueError('Game representation not supported!')
