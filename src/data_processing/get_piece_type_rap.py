import glob
import os
from os import path
import re
import numpy as np
import random
import argparse
import chess


def convert_uci_to_san(uci_moves):
    san_moves = []
    board = chess.Board()
    for uci_move in uci_moves:
        move = board.parse_uci(uci_move)
        san_moves.append(board.san(move))
        board.push(move)

    return san_moves


def main(args):
    random.seed(42)
    np.random.seed(42)

    output_dir = path.join(args.data_dir, "rap")
    print(f"Output directory: {output_dir}")

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    uci_dir = path.join(args.data_dir, "uci")

    files = [f for f in glob.glob(path.join(uci_dir, '*.txt')) if re.match('.*/([a-z]*).txt', f)]
    print("Source files:", files)

    for split_file in files:
        all_piece_type_list = []
        basename = path.splitext(path.basename(split_file))[0]
        output_file = path.join(output_dir, basename + ".npy")

        with open(split_file) as g:

            for uci_line in g:
                uci_moves = uci_line.strip().split()
                san_moves = convert_uci_to_san(uci_moves)

                piece_type_list = []
                for san_move in san_moves:
                    if san_move[0] != 'O':
                        if san_move[0].isupper():
                            piece_type_list.append(san_move[0])
                        else:
                            piece_type_list.append('P')
                    else:
                        piece_type_list.append(None)

                all_piece_type_list.append(piece_type_list)

        np.save(output_file, all_piece_type_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="Base directory with sub-directories corresponding to notation types.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())


