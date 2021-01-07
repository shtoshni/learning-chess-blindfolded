import json
import argparse
import os
from os import path
import numpy as np

from constants import LENGTH_CATEGORIES
from data_processing.get_fen_repr import game_to_parsed_fen_sequence


def get_fen_for_file(input_dir, category, output_dir):
    input_file = path.join(input_dir, f'end_{category}.jsonl')
    basename = f'other_eval_{category}'
    output_file = path.join(output_dir, basename + ".npy")

    all_games = []
    with open(input_file) as reader:
        for idx, line in enumerate(reader):
            if (idx + 1) % 1000 == 0:
                print(f"{idx + 1}th game")
            game = json.loads(line.strip())
            move_seq = game["prefix"].split()
            game_array = game_to_parsed_fen_sequence(move_seq)
            all_games.append(game_array)

    np.save(output_file, all_games)


def get_fen_for_dir(input_dir, output_dir):
    for category in LENGTH_CATEGORIES:
        print(category)
        get_fen_for_file(input_dir, category, output_dir)


def main():
    args = parse_args()
    uci_dir = path.join(args.data_dir, 'other_eval/uci')
    output_dir = path.join(args.data_dir, 'fen')

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    get_fen_for_dir(input_dir=uci_dir, output_dir=output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str,
                        help="Data directory where sub-directories correspond to notation-level directories.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
