import chess
import os
from os import path
import argparse
import glob
import json

from collections import OrderedDict, defaultdict
from prettytable import PrettyTable


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
    is_check = board.is_check()
    if is_check:
        for pseudo_legal_move in board.pseudo_legal_moves:
            if pred == pseudo_legal_move.uci()[2:]:
                # Pseudo legal move
                return "Pseudo Legal"

    return "Path Obstruction"


def analyze_file(input_file):
    error_counter = {"actual": defaultdict(list), "other": defaultdict(list)}
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
                    # counter[category] += 1
                    error_type = analyze_error(board_state, piece_type, starting_square, pred)
                    error_counter[category][error_type].append(instance)

    return error_counter


def analyze(input_dir, output_dir):
    actual_files = sorted(glob.glob(path.join(input_dir, f'end_*')))

    error_categories = ['Syntax', 'Path Obstruction', 'Pseudo Legal']

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

        stats.add_row(error_counts)
        print(output_file)
        print(' & '.join(error_counts))
        print(stats)


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


