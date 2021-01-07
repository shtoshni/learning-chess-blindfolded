import chess
import os
from os import path
import argparse
import glob
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_dir', type=str, help='Model directory')
    parser.add_argument('-base_analysis_dir', type=str, default='../models/analysis',
                        help='Analysis directory')

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

    try:
        # print(prefix)
        prefix = [move[1:] if move[0].isupper() else move for move in prefix]
        move_prefix = move_prefix[1:] if move_prefix[0].isupper() else move_prefix

        board = chess.Board()

        for move in prefix:
            board.push_uci(move)
    except ValueError:
        # print(prefix)
        import sys
        sys.exit()

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
    if mode == 'actual':
        # Ignore 2nd and 3rd column
        instance = [instance[0]] + instance[3:]

    prompt = instance[0].split()
    game_prefix, move_prefix = prompt[:-1], prompt[-1]

    legal_corr = int(instance[2])
    syntax_corr = int(instance[4])
    pred = instance[5]

    info_dict = {'game_prefix': game_prefix, 'move_prefix': move_prefix,
                 'legal_corr': legal_corr, 'legal_options': instance[1].split(),
                 'syntax_corr': syntax_corr, 'syntax_options': instance[3].split(),
                 'pred': pred}

    return info_dict


def analyze_file(output_dir, file_path):
    output_file = path.join(output_dir, path.basename(file_path))
    lines = [line.strip() for line in open(file_path).readlines()]
    if len(lines[0].split(",")) > 6:
        mode = 'actual'
    else:
        mode = 'other'

    legal = {}
    for result in ['correct', 'incorrect']:
        legal[result] = OrderedDict(
            [('counter', 0), ('piece_type', OrderedDict([('R', 0), ('B', 0), ('N', 0), ('Q', 0), ('K', 0)])),
             ('gap', []), ('prefix_size', []),
             # ('occupied', 0), ('occupied_initial', 0),
             ('path_obstruct', 0),
             ('pseudo_legal', 0), ('syntax_err', 0),
             ('is_check', defaultdict(int)), ('never_moved', 0)])

    with open(output_file, 'w') as writer:
        for line in lines[1:]:
            parsed_dict = parse_line(line, mode=mode)
            prompt_dict = analyze_prompt(parsed_dict['game_prefix'], parsed_dict['move_prefix'])

            parsed_dict.update(prompt_dict)
            result = 'incorrect'
            if parsed_dict['legal_corr']:
                result = 'correct'

            board = parsed_dict['board']
            game_prefix = parsed_dict['game_prefix']
            prediction = parsed_dict['pred']
            move_prefix = parsed_dict['move_prefix']

            if result == 'incorrect':
                if parsed_dict['syntax_corr']:
                    # Legal mistake but syntactically fine
                    pred_move = move_prefix + prediction
                    pseudo_legal_moves = set()
                    for pseudo_legal_move in board.pseudo_legal_moves:
                        pseudo_legal_moves.add(board.uci(pseudo_legal_move))

                    if pred_move in pseudo_legal_moves:
                        # The move places king in check
                        error_type = 'pseudo_legal'
                    else:
                        error_type = 'path_obstruct'
                else:
                    error_type = 'syntax_err'

                legal[result][error_type] += 1

                writer.write(f'{" ".join(game_prefix)}\n')
                writer.write(f'{parsed_dict["piece_type"]}, {move_prefix}, {prediction}, {error_type}\n')

            legal[result]['piece_type'][parsed_dict['piece_type']] += 1
            legal[result]['is_check'][parsed_dict['is_check']] += 1
            legal[result]['prefix_size'].append(parsed_dict['prefix_size'])
            if parsed_dict['last_moved_idx'] is not None:
                legal[result]['gap'].append(parsed_dict['prefix_size'] - parsed_dict['last_moved_idx'])
            else:
                legal[result]['never_moved'] += 1

            legal[result]['counter'] += 1

        for result in ['correct', 'incorrect']:
            for field in ['prefix_size', 'gap']:
                legal[result][field] = round(sum(legal[result][field])/(len(legal[result][field]) + 1e-6), 2)

            for piece_type in legal[result]['piece_type']:
                legal[result]['piece_type'][piece_type] = round(
                    legal[result]['piece_type'][piece_type]/legal[result]['counter'], 2)

    # print()
    # print(path.basename(file_path))
    # print(legal['correct'])
    # print(legal['incorrect'])
    eval_type = ' '.join([piece.capitalize() for piece in path.basename(file_path).split(".")[0].split("_")[1:]])
    # print(eval_type)
    output_dict = OrderedDict()
    output_dict['eval_type'] = eval_type
    output_dict['Errors (in 1k)'] = legal['incorrect']['counter']
    output_dict['Path Obstruction'] = legal['incorrect']['path_obstruct']
    output_dict['Syntax'] = legal['incorrect']['syntax_err']
    output_dict['Pseudo Legal'] = legal['incorrect']['pseudo_legal']
    return output_dict


def analyze(input_dir, output_dir):
    # for category in ['human', 'synthetic']:
    for category in ['human']:
        print(category.capitalize())
        actual_files = sorted(glob.glob(path.join(input_dir, f'{category}_end_*')))

        print(output_dir)
        from prettytable import PrettyTable
        stats = PrettyTable(['Eval Type', 'Errors (in 1k)', 'Path Obstruction', 'Syntax', 'Pseudo Legal'])

        total_dict = defaultdict(int)
        for file in actual_files:
            output_dict = analyze_file(output_dir, file)
            for key, val in output_dict.items():
                if key != 'eval_type':
                    total_dict[key] += val

            stats.add_row([output_dict['eval_type'], output_dict['Errors (in 1k)'],  output_dict['Path Obstruction'],
                           output_dict['Syntax'], output_dict['Pseudo Legal']])

        stats.add_row(['Total (4k)', total_dict['Errors (in 1k)'], total_dict['Path Obstruction'],
                       total_dict['Syntax'], total_dict['Pseudo Legal']])
        print(stats)


def main(args):
    other_eval_dir = path.join(args.model_dir, 'other_eval')
    model_name = path.basename(args.model_dir.rstrip("/"))

    analysis_dir = path.join(args.base_analysis_dir, model_name)
    if not path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    analyze(other_eval_dir, analysis_dir)


if __name__ == '__main__':
    main(parse_args())


