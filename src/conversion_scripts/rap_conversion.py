import glob
import os
from os import path
import re
import numpy as np
import random
import argparse
from chess_utils.conversion_utils import convert_lan_to_uci


def main(args):
    random.seed(42)
    np.random.seed(42)

    output_dir = path.join(args.base_dir, f"rap_{int(args.random_prob * 100)}")
    print(output_dir)

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    lan_dir = path.join(args.base_dir, "lan_with_p")
    uci_dir = path.join(args.base_dir, "uci")

    files = [f for f in glob.glob(path.join(lan_dir, '*.txt')) if re.match('.*/([a-z]*|train.*).txt', f)]
    print("Source files:", files)

    for lan_file in files:
        basename = path.basename(lan_file)
        uci_file = path.join(uci_dir, basename)

        output_file = path.join(output_dir, basename)

        with open(lan_file) as f, open(uci_file) as g, open(output_file, 'w') as writer:
            for lan_line, uci_line in zip(f, g):
                lan_moves = lan_line.strip().split()
                uci_moves = uci_line.strip().split()

                assert (uci_moves == convert_lan_to_uci(lan_moves))
                random_vec = np.random.choice([0, 1], size=(len(uci_moves),), p=[1 - args.random_prob, args.random_prob])

                rap_moves = []
                for idx, (uci_move, lan_move) in enumerate(zip(uci_moves, lan_moves)):
                    if random_vec[idx] == 1 and lan_move[0] != 'O':
                        rap_moves.append(lan_move[0] + uci_move)
                    else:
                        rap_moves.append(uci_move)

                writer.write(' '.join(rap_moves) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        help="Base directory with sub-directories corresponding to notation types.")
    parser.add_argument("--random_prob", default=0.15, type=float, help="Piece type probability.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())


