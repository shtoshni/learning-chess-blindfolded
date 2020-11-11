import argparse
import random
import os
from os import path

from chess_utils.conversion_utils import convert_move_to_notation, convert_game_to_notation
from chess_utils.utils import get_move_prefix_size, get_syntax_moves, get_syntax_locations
from collections import OrderedDict, defaultdict

PREFIX_SIZE_TO_RANGE = {'easy': (5, 50), 'hard': (51, 100)}
SEP = ","


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-base_dir', type=str, default=None,
                        help="Base data directory where all notation specific directories are present.")
    parser.add_argument('-max_games', type=int, default=1000,
                        help='Number of games per eval type')

    args = parser.parse_args()

    assert(path.exists(args.base_dir))
    args.data_dir = path.join(args.base_dir, "lan")
    assert (path.exists(args.data_dir))

    args.base_output_dir = path.join(args.base_dir, "other_eval")
    if not path.exists(args.base_output_dir):
        os.makedirs(args.base_output_dir)

    args.output_dir_dict = {
        "lan": path.join(args.base_output_dir, "lan_with_p"),
        "san": path.join(args.base_output_dir, "san"),
        "uci": path.join(args.base_output_dir, "uci"),
    }

    for notation, output_dir in args.output_dir_dict.items():
        if not path.exists(output_dir):
            os.makedirs(output_dir)

    return args


def load_train_prefixes(args):
    prefix_hash_set = set()
    train_file = path.join(args.data_dir, "train_medium.txt")

    with open(train_file) as f:
        for line in f:
            moves = line.strip().split()
            for i in range(1, len(moves)):
                prefix_hash_set.add(hash(tuple(moves[0: i])))

    print(f"Unioue train prefixes: {len(prefix_hash_set)}")
    return prefix_hash_set


def get_eval_prefixes(args, train_prefix_set):
    random.seed(42)
    data_file = path.join(args.data_dir, "other_eval.txt")

    eval_prefixes = []
    with open(data_file) as f:
        game_counter = 0
        for idx, line in enumerate(f):
            moves = line.strip().split()
            num_moves = len(moves)

            if num_moves <= args.min_size:
                continue

            prefix_size = random.randint(args.min_size, min(num_moves - 1, args.max_size))
            game_prefix = moves[:prefix_size]
            prefix_hash = hash(tuple(game_prefix))
            if prefix_hash in train_prefix_set:
                continue

            actual_move = moves[prefix_size]
            if actual_move[0] != 'O' and actual_move[0].isupper():
                # Only consider moves made by pieces other than pawns which don't involve castling
                eval_prefixes.append((game_prefix, actual_move))
                game_counter += 1

            if game_counter >= args.max_games:
                break

    return eval_prefixes


def get_end_position_eval_data(args, eval_prefixes):
    random.seed(42)

    for mode in ["actual", "other"]:
        print(f"\nMode {mode.capitalize()}")
        output_file = f"dev_end_{mode}_{args.prefix_size}.txt"

        lan_choices = []

        for notation in ["lan", "uci"]:
            output_dir = args.output_dir_dict[notation]
            eval_output_file = path.join(output_dir, output_file)

            max_options = defaultdict(int)

            print(f"Creating eval set for {notation.upper()}")

            with open(eval_output_file, 'w') as writer:
                if mode == "actual":
                    writer.write(f"prefix{SEP}actual{SEP}legal{SEP}syntax\n")
                else:
                    writer.write(f"prefix{SEP}legal{SEP}syntax\n")

                for idx, (game_prefix, actual_move) in enumerate(eval_prefixes):
                    prefix_moves, board = convert_game_to_notation(game_prefix, notation=notation)
                    actual_move = convert_move_to_notation(actual_move, board, notation=notation)

                    move_prefix_size = get_move_prefix_size(actual_move, notation=notation)
                    actual_move_prefix, actual_move_suffix = (actual_move[:move_prefix_size],
                                                              actual_move[move_prefix_size:])

                    legal_move_prefix = OrderedDict()
                    for legal_move in board.legal_moves:
                        # print(bo)
                        san_move = board.san(legal_move)
                        if san_move[0].isupper():
                            if san_move[0] == 'O' and notation != 'uci':
                                # For castling, except for UCI, other notations won't have prefixes with piece names
                                continue
                            move_repr = convert_move_to_notation(legal_move, board, notation=notation)
                            move_prefix_size = get_move_prefix_size(move_repr, notation=notation)

                            move_prefix, move_suffix = move_repr[:move_prefix_size], move_repr[move_prefix_size:]
                            if move_prefix in legal_move_prefix:
                                legal_move_prefix[move_prefix].add(move_suffix)
                            else:
                                legal_move_prefix[move_prefix] = {move_suffix}

                    if mode == "actual":
                        selected_move_prefix = actual_move_prefix
                        actual_move_options = [actual_move_suffix]
                        legal_move_options = legal_move_prefix[actual_move_prefix]

                        assert (set(actual_move_options).issubset(set(legal_move_options)))
                    else:
                        if notation == "lan":
                            selected_move_prefix = random.choice(list(legal_move_prefix.keys()))
                            piece, square = selected_move_prefix[0], selected_move_prefix[1:3]
                            lan_choices.append({"uci": square, "lan": move_prefix})
                        elif notation == "uci":
                            selected_move_prefix = lan_choices[idx][notation]
                        else:
                            raise ValueError

                        legal_move_options = legal_move_prefix[selected_move_prefix]

                    # Next add syntax options
                    if notation == "lan":
                        piece_location = selected_move_prefix[1:3]
                    elif notation == 'uci':
                        piece_location = selected_move_prefix[:2]
                    else:
                        raise ValueError

                    syntax_move_options = get_syntax_moves(board, piece_location, notation=notation)
                    # Removing this Assertion because for UCI, king castling is a legal move but not a syntax move
                    # if notation == "uci":
                    #     assert (set(legal_move_options).issubset(set(syntax_move_options)))
                    # else:
                    syntax_move_options = list(set(syntax_move_options + list(legal_move_options)))

                    instance_string = (" ".join(prefix_moves) + " " + selected_move_prefix
                                       + (SEP + " ".join(actual_move_options) if mode == "actual" else '')
                                       + SEP + " ".join(legal_move_options)
                                       + SEP + " ".join(syntax_move_options))

                    writer.write(instance_string + "\n")
                    max_options['legal'] = max(max_options['legal'], len(legal_move_options))
                    max_options['syntax'] = max(max_options['syntax'], len(syntax_move_options))

            print("Max options:", max_options)


def get_start_position_eval_data(args, eval_prefixes):
    random.seed(42)
    notation = "lan"

    for mode in ["actual", "other"]:
        print(f"Mode {mode.capitalize()}")

        output_dir = args.output_dir_dict[notation]
        output_file = f"dev_start_{mode}_{args.prefix_size}.txt"
        eval_output_file = path.join(output_dir, output_file)

        max_options = defaultdict(int)
        print(f"Creating eval set for LAN")
        with open(eval_output_file, 'w') as writer:
            if mode == "actual":
                writer.write(f"prefix{SEP}actual{SEP}legal{SEP}syntax\n")
            else:
                writer.write(f"prefix{SEP}legal{SEP}syntax\n")

            for idx, (game_prefix, move) in enumerate(eval_prefixes):
                prefix_moves, board = convert_game_to_notation(game_prefix, notation="lan")
                actual_move = convert_move_to_notation(move, board, notation="lan")

                legal_move_prefix = OrderedDict()
                for legal_move in board.legal_moves:
                    san_move = board.san(legal_move)
                    if san_move[0] != 'O' and san_move[0].isupper():
                        move_repr = convert_move_to_notation(legal_move, board, notation="lan")
                        piece_moved, piece_location = move_repr[0], move_repr[1:3]
                        if piece_moved in legal_move_prefix:
                            legal_move_prefix[piece_moved].add(piece_location)
                        else:
                            legal_move_prefix[piece_moved] = {piece_location}

                if mode == 'actual':
                    actual_piece_locations = [actual_move[1:3]]
                    legal_piece_locations = legal_move_prefix[actual_move[0]]

                    piece_symbol = actual_move[0]
                    piece_symbol = piece_symbol if (len(prefix_moves) % 2 == 0) else piece_symbol.lower()
                    syntax_piece_locations = get_syntax_locations(board, piece_symbol)

                    assert(set(actual_piece_locations).issubset(set(legal_piece_locations)))
                else:
                    piece_symbol = random.choice(list(legal_move_prefix.keys()))
                    legal_piece_locations = legal_move_prefix[piece_symbol]

                    piece_symbol = piece_symbol if (len(prefix_moves) % 2 == 0) else piece_symbol.lower()
                    syntax_piece_locations = get_syntax_locations(board, piece_symbol)

                try:
                    assert(set(legal_piece_locations).issubset(set(syntax_piece_locations)))
                except AssertionError:
                    print(legal_piece_locations, syntax_piece_locations)
                    print(len(prefix_moves))
                    print()

                instance_string = (" ".join(prefix_moves) + " " + piece_symbol.upper()
                                   + (SEP + " ".join(actual_piece_locations) if mode == 'actual' else '')
                                   + SEP + " ".join(legal_piece_locations)
                                   + SEP + " ".join(syntax_piece_locations))

                writer.write(instance_string + "\n")
                max_options['legal'] = max(max_options['legal'], len(legal_piece_locations))
                max_options['syntax'] = max(max_options['syntax'], len(syntax_piece_locations))

        print("Max options:", max_options)


def main():
    args = parse_args()
    train_prefix_set = load_train_prefixes(args)

    for prefix_size in PREFIX_SIZE_TO_RANGE:
        print(f"\nGenerating eval set for {prefix_size}\n")
        args.prefix_size = prefix_size
        args.min_size, args.max_size = PREFIX_SIZE_TO_RANGE[args.prefix_size]
        eval_prefixes = get_eval_prefixes(args, train_prefix_set)
        print("Getting START position eval set")
        get_start_position_eval_data(args, eval_prefixes)
        print("Getting END position eval set")
        get_end_position_eval_data(args, eval_prefixes)


if __name__ == '__main__':
    main()
