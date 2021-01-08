import argparse
import os
import logging
import json
import chess
import chess.pgn as pgn
import bz2
import glob

from os import path
from collections import OrderedDict
from contextlib import suppress

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str, help="PGN dir")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--max_games", default=10_000_000, type=int, help="Max games to parse")

    parsed_args = parser.parse_args()
    assert (path.exists(parsed_args.source_file))
    if not path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)

    return parsed_args


def parse_pgn(args):
    basename = path.splitext(path.basename(args.source_file))[0]
    output_file = path.join(args.output_dir, f"{basename}_uci.txt")

    game_counter = 0
    with open(output_file, 'w') as output_f, open(args.source_file) as source_f:
        while True:
            try:
                game = pgn.read_game(source_f)
                board = game.board()
                if game is None:
                    break
                move_list = []
                for move in game.mainline_moves():
                    move_list.append(board.uci(move))
                    board.push(move)

                output_f.write(" ".join(move_list) + "\n")

                game_counter += 1
                if game_counter % 1e4 == 0:
                    logger.info(f"{game_counter} games processed")

                if game_counter >= args.max_games:
                    break
            except (ValueError, UnicodeDecodeError) as e:
                pass

    logger.info(f"Extracted {game_counter} games at {output_file}")


if __name__ == '__main__':
    parse_pgn(parse_args())
