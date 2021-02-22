import chess
import os
from os import path
import argparse
import glob
import json

from collections import OrderedDict, defaultdict
from prettytable import PrettyTable
from copy import deepcopy


def get_board_state(prefix):
    if isinstance(prefix, str):
        prefix = prefix.strip().split()

    board = chess.Board()

    for move in prefix:
        if move[0].isupper():
            move = move[1:]
        board.push_uci(move)

    return board


def analyze_error(board, piece_type, starting_square, pred):
    square_idx = chess.SQUARE_NAMES.index(starting_square)

    # Check for spatial errors first
    spatial_error = True
    for cur_piece_type in chess.PIECE_SYMBOLS[1:]:
        if cur_piece_type == piece_type:
            continue
        # board_copy = deepcopy(board)
        board_copy = chess.Board()
        board_copy.clear_board()
        board_copy._set_piece_at(
            square_idx, chess.PIECE_SYMBOLS.index(cur_piece_type.lower()), color=board_copy.turn)

        legal_endings = []
        for move in board_copy.legal_moves:
            legal_endings.append(move.uci()[2:])

        if pred in legal_endings:
            # Some other piece can do that
            spatial_error = False
            break

    if spatial_error:
        return "Spatial"

    # Check for syntax errors first
    empty_board = chess.Board()
    empty_board.clear_board()
    square_idx = chess.SQUARE_NAMES.index(starting_square)
    empty_board._set_piece_at(square_idx, chess.PIECE_SYMBOLS.index(piece_type.lower()), color=bool(1))

    legal_endings = []
    for move in empty_board.legal_moves:
        legal_endings.append(move.uci()[2:])

    if pred not in legal_endings:
        return "Syntax"

    # Pseudo-legal check
    # The king remains in check after the move is executed
    # Or the move introduces king to check
    for pseudo_legal_move in board.pseudo_legal_moves:
        if (starting_square + pred) == pseudo_legal_move.uci():
            # Pseudo legal move
            return "Pseudo Legal"

    return "Path Obstruction"


def analyze_file(input_file):
    error_counter = {"actual": defaultdict(list), "other": defaultdict(list),
                     "correct-actual": [], "correct-other": []}
    with open(input_file) as reader:
        for line in reader:
            instance = json.loads(line.strip())
            board_state = get_board_state(instance["prefix"])

            for category, prefix in zip(["actual", "other"], ["", "other_"]):
                result = instance[category]["LgM"]
                piece_type = instance[prefix + "piece_type"]
                starting_square = instance[prefix + "starting_square"]
                pred = result["pred"][0]

                if result["corr"] != 1:
                    error_type = analyze_error(board_state, piece_type, starting_square, pred)
                    error_counter[category][error_type].append(instance)
                else:
                    error_counter[f"correct-{category}"].append(instance)

    return error_counter


def analyze(input_dir, output_dir):
    actual_files = sorted(glob.glob(path.join(input_dir, f'end_long*')))

    error_categories = ['Spatial', 'Syntax', 'Path Obstruction', 'Pseudo Legal']

    for log_file in actual_files:
        stats = PrettyTable(error_categories + ['Other ' + category for category in error_categories])
        error_dict = analyze_file(log_file)

        output_file = path.join(output_dir, path.basename(log_file))
        with open(output_file, 'w') as writer:
            error_counts = []

            for category in ["actual", "other"]:
                for error_category in error_categories:
                    error_counts.append(str(len(error_dict[category][error_category])))

                    output_dict = OrderedDict()
                    output_dict[category + "-" + error_category] = error_dict[category][error_category]
                    writer.write(json.dumps(output_dict) + "\n")

                output_dict = {f'correct-{category}': error_dict[f'correct-{category}']}
                writer.write(json.dumps(output_dict) + "\n")

        stats.add_row(error_counts)
        print(output_file)

        print(stats)
        # Skip the spatial ones in the latex code
        error_counts = error_counts[1:4] + error_counts[5:]
        output_str = ""
        for idx, error_count in enumerate(error_counts):
            if idx in [1, 2, 5]:
                # Path obstruction or pseudo legal
                if int(error_count) < 10:
                    output_str += '\phantom{1}'
            else:
                # Syntax error
                if int(error_count) < 10:
                    output_str += '\phantom{11}'
                elif int(error_count) < 100:
                    output_str += '\phantom{1}'
            output_str += f'{error_count} & '

        print("\n\n" + output_str.strip('& ') + "\n\n")


def main(args):
    other_eval_dir = path.join(args.model_dir, 'other_eval')
    model_name = path.basename(args.model_dir.rstrip("/"))

    analysis_dir = path.join(args.base_analysis_dir, model_name)
    if not path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    analyze(other_eval_dir, analysis_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, help='Model directory')
    parser.add_argument('--base_analysis_dir', type=str, default='/tmp',
                        help='Analysis directory')

    args = parser.parse_args()
    assert (path.exists(args.model_dir))

    return args


if __name__ == '__main__':
    main(parse_args())


