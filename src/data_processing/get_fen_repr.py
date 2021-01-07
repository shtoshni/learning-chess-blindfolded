import chess
import argparse
import os
import glob

from os import path
import numpy as np
import re

from constants import PIECE_TO_VAL


def parse_fen(fen):
    fen_split = fen.split()
    board = fen_split[0]
    fen_arr = []
    for char in board:
        if char.isdigit():
            fen_arr += [-1] * int(char)
        elif char != '/':
            fen_arr.append(PIECE_TO_VAL[char])

    castling_rights = fen_split[2]
    castling_bits = [0, 0, 0, 0]
    if 'K' in castling_rights:
        castling_bits[0] = 1
    if 'Q' in castling_rights:
        castling_bits[1] = 1
    if 'k' in castling_rights:
        castling_bits[2] = 1
    if 'q' in castling_rights:
        castling_bits[3] = 1

    fen_arr.extend(castling_bits)
    return fen_arr


def three_way_fen_coding(fen):
    fen_arr = parse_fen(fen)

    three_way_repr = np.zeros((64, 6))
    for idx, fen_val in enumerate(fen_arr):
        if fen_val > -1:
            piece_type = fen_val // 2
            parity = (1 if fen_val % 2 else -1)

            three_way_repr[idx, piece_type] = parity

    return three_way_repr


def initialize_game_array():
    board = chess.Board()
    fen = board.fen()
    parsed_fen = parse_fen(fen)

    is_check = int(board.is_check())
    white_or_black = 1
    parsed_fen.extend([is_check, white_or_black])

    game_array = np.array(parsed_fen, dtype=np.int8)
    return game_array


def game_to_parsed_fen_sequence(move_seq):
    board = chess.Board()
    game_array = initialize_game_array()

    for idx, move in enumerate(move_seq):
        board.push_uci(move)
        fen = board.fen()
        parsed_fen = parse_fen(fen)

        is_check = int(board.is_check())
        white_or_black = idx % 2

        parsed_fen.extend([is_check, white_or_black])
        parsed_fen = np.array(parsed_fen, dtype=np.int8)
        game_array = np.append(game_array, parsed_fen)

    return game_array


def get_fen_for_file(input_file, output_dir):
    basename = path.splitext(path.basename(input_file))[0]
    output_file = path.join(output_dir, basename + ".npy")

    all_games = []
    with open(input_file) as reader:
        for idx, line in enumerate(reader):
            if (idx + 1) % 1000 == 0:
                print(f"{idx + 1}th game")
            move_seq = line.strip().split()
            game_array = game_to_parsed_fen_sequence(move_seq)
            all_games.append(game_array)

    np.save(output_file, all_games)


def get_fen_for_dir(input_dir, output_dir):
    input_files = [f for f in glob.glob(path.join(input_dir, '*.txt')) if re.match('.*/([a-z]*).txt', f)]
    print(input_files)
    for input_file in input_files:
        print(input_file)
        get_fen_for_file(input_file, output_dir)


def main():
    args = parse_args()
    uci_dir = path.join(args.data_dir, 'uci')
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

