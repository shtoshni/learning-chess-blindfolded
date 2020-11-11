import chess
import os
from os import path
import argparse
import glob
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_dir', type=str, help='Model directory')
    args = parser.parse_args()
    assert (path.exists(args.model_dir))

    return args


def get_last_move_idx(prefix, move_prefix):
    last_moved_idx = None
    prefix_move_squares = [(move[:2], move[2:4]) for move in prefix]
    for idx, (_, move_end) in enumerate(prefix_move_squares[-2::-2]):
        if move_end == move_prefix:
            last_moved_idx = (idx + 1) * (-2) + len(prefix) + 1
    return last_moved_idx


def analyze_prompt(prefix, move_prefix):
    if isinstance(prefix, str):
        prefix = prefix.strip().split()

    prefix = [move[1:5] if move[0].isupper() else move for move in prefix]
    move_prefix = move_prefix[1:] if move_prefix[0].isupper() else move_prefix

    board = chess.Board()

    for move in prefix:
        board.push_uci(move)

    is_check = board.is_check()

    color = 'white' if (len(prefix) % 2 == 0) else 'black'

    square_idx = chess.SQUARE_NAMES.index(move_prefix)
    piece_type = str(board.piece_at(square_idx)).upper()
    last_moved_idx = get_last_move_idx(prefix, move_prefix)

    return {'piece_type': piece_type, 'last_moved_idx': last_moved_idx,
            'is_check': is_check, 'prefix_size': len(prefix), 'color': color,
            'board': board, 'game_prefix': prefix, 'move_prefix': move_prefix}


def parse_line(line, mode='actual'):
    instance = line.split(",")
    info_dict = OrderedDict()
    if mode == 'actual':
        # Ignore 2nd and 3rd column
        info_dict['AM'] = int(instance[2])
        instance = [instance[0]] + instance[3:]

    # prompt = instance[0].split()
    # game_prefix, move_prefix = prompt[:-1], prompt[-1]

    info_dict['LM'] = int(instance[2])
    info_dict['SM'] = int(instance[4])

    return info_dict


def analyze_file(file_path):
    lines = [line.strip() for line in open(file_path).readlines()]
    if len(lines[0].split(",")) > 6:
        mode = 'actual'
    else:
        mode = 'other'

    output_dict = OrderedDict([('counter', 0), ('AM', 0), ('LM', 0), ('SM', 0)])

    for line in lines[1:]:
        info_dict = parse_line(line, mode=mode)

        for key, val in info_dict.items():
            output_dict[key] += val

        output_dict['counter'] += 1

    for key, val in output_dict.items():
        if key != 'counter':
            output_dict[key] = val * 100.0/output_dict['counter']

    return output_dict


def analyze(input_dir):
    # for category in ['human', 'synthetic']:
    latex_str = ''
    for category in ['human']:
        print(category.capitalize())
        actual_files = [path.join(input_dir, f'{category}_end_actual_easy.log'),
                        path.join(input_dir, f'{category}_end_other_easy.log'),
                        path.join(input_dir, f'{category}_end_actual_hard.log'),
                        path.join(input_dir, f'{category}_end_other_hard.log')
                        ]
        for file in actual_files:
            output_dict = analyze_file(file)
            print(output_dict)
            if output_dict['AM']:
                latex_str += f"& {output_dict['AM']} & {output_dict['LM']}"
            else:
                latex_str += f"& {output_dict['LM']}"

    print(latex_str)
        # from prettytable import PrettyTable
        # stats = PrettyTable(['Eval Type', 'Errors (in 1k)', 'Path Obstruction', 'Syntax', 'Pseudo Legal'])
        #
        # total_dict = defaultdict(int)
        # for file in actual_files:
        #     output_dict = analyze_file(file)
        #     for key, val in output_dict.items():
        #         if key != 'eval_type':
        #             total_dict[key] += val
        #
        #     stats.add_row([output_dict['eval_type'], output_dict['Errors (in 1k)'],  output_dict['Path Obstruction'],
        #                    output_dict['Syntax'], output_dict['Pseudo Legal']])
        #
        # stats.add_row(['Total (4k)', total_dict['Errors (in 1k)'], total_dict['Path Obstruction'],
        #                total_dict['Syntax'], total_dict['Pseudo Legal']])
        # print(stats)


def main(args):
    other_eval_dir = path.join(args.model_dir, 'other_eval')
    model_name = path.basename(args.model_dir.rstrip("/"))

    analyze(other_eval_dir)


if __name__ == '__main__':
    main(parse_args())


