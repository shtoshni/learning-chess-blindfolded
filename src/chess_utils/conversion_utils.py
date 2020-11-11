import chess


def convert_lan_to_uci(game_lan):
    """Convert game to UCI notation."""
    uci_notation = []
    for move_idx, move in enumerate(game_lan):
        if 'O-O' in move:
            first_player = False
            if move_idx % 2 == 0:
                first_player = True

            if move == 'O-O' or move == 'O-O+':
                if first_player:
                    uci_repr = 'e1g1'
                else:
                    uci_repr = 'e8g8'
            else:
                if first_player:
                    uci_repr = 'e1c1'
                else:
                    uci_repr = 'e8c8'

            uci_notation.append(uci_repr)
        else:
            sep_count = move.count('x') + move.count('-')
            if sep_count != 1:
                break

            first_part, second_part = None, None
            if 'x' in move:
                first_part, second_part = move.split('x')
            elif '-' in move:
                first_part, second_part = move.split('-')

            move_notation = first_part[-2:] + second_part[:2]
            if '=' in move:
                prom_index = move.index('=') + 1
                piece_name = move[prom_index].lower()
                move_notation += piece_name

            uci_notation.append(move_notation)

    return uci_notation


def convert_lan_to_san(game_lan):
    game_uci = convert_lan_to_uci(game_lan)
    return convert_game_notation(game_uci, target_notation="san", source_notation="uci")


def convert_lan_to_lan_with_p(game_lan):
    return ['P' + move if not move[0].isupper() else move for move in game_lan]


def convert_game_notation(game, target_notation, source_notation=None):
    if source_notation is None:
        pass

    if source_notation == 'lan':
        if target_notation == 'uci':
            return convert_lan_to_uci(game)
        elif target_notation == 'san':
            return convert_lan_to_san(game)
        elif target_notation == "lan_with_p":
            return convert_lan_to_lan_with_p(game)

    elif (source_notation in ['uci', 'san']) and (target_notation in ['san', 'uci', 'lan']):
        board = chess.Board()
        parse_fn = getattr(board, 'parse_' + source_notation)
        conversion_fn = getattr(board, target_notation)

        converted_game = []
        for move in game:
            parsed_move = parse_fn(move)
            converted_game.append(conversion_fn(parsed_move))
            board.push(parsed_move)

        return converted_game

    elif ('rap' in source_notation) and (target_notation == 'uci'):
        return [move[1:] if move[0].isupper() else move for move in game]

    else:
        raise ValueError(f"Conversion between {source_notation} and {target_notation} not supported")


def convert_move_to_lan_plus(move, board, add_p=True):
    if isinstance(move, str):
        move = board.parse_san(move)
    lan_move = board.lan(move)
    if not lan_move[0].isupper():
        lan_move = 'P' + lan_move
    return lan_move


def convert_move_to_san(move, board):
    if isinstance(move, str):
        return move
    else:
        return board.san(move)


def convert_move_to_uci(move, board):
    if isinstance(move, str):
        move = board.parse_san(move)
    return board.uci(move)


def convert_move_to_notation(move, board, notation='lan'):
    if notation == 'lan':
        return convert_move_to_lan_plus(move, board)
    elif notation == 'san':
        return convert_move_to_san(move, board)
    elif notation == 'uci':
        return convert_move_to_uci(move, board)


def convert_game_to_lan_plus(game, add_p=True):
    board = chess.Board()
    moves = []
    for move in game:
        san_move = board.parse_san(move)
        moves.append(board.lan(san_move))
        board.push(san_move)
    if not add_p:
        return moves, board
    else:
        return ['P' + move if not move[0].isupper() else move for move in moves], board


def convert_game_to_san(game):
    board = chess.Board()
    moves = []
    for move in game:
        san_move = board.parse_san(move)
        moves.append(board.san(san_move))
        board.push(san_move)
    return moves, board


def convert_game_to_uci(game):
    board = chess.Board()
    uci_moves = []
    for move in game:
        san_move = board.parse_san(move)
        uci_moves.append(board.uci(san_move))
        board.push(san_move)
    return uci_moves, board


def convert_game_to_notation(game, notation='lan'):
    if notation == 'lan':
        return convert_game_to_lan_plus(game)
    elif notation == 'san':
        return convert_game_to_san(game)
    elif notation == 'uci':
        return convert_game_to_uci(game)