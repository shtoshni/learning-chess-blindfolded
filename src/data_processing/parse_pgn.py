import chess
import chess.pgn as pgn

import argparse
from os import path
import os
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str, help="PGN file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--max_games", default=1e8, type=int, help="Max games to parse")

    parsed_args = parser.parse_args()
    assert(path.exists(parsed_args.source_file))
    if not path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)

    return parsed_args


def parse_pgn(args):
    basename = path.splitext(path.basename(args.source_file))[0]
    san_file = path.join(args.output_dir, f"{basename}_san.txt")

    game_counter = 0
    with open(san_file, 'w') as san_f:
        while True:
            try:
                game = pgn.read_game(open(args.source_file))
                board = game.board()
                if game is None:
                    break
                san_list, lan_list, uci_list = [], [], []
                for move in game.mainline_moves():
                    san_list.append(board.san(move))
                    board.push(move)

                san_f.write(" ".join(san_list) + "\n")

                game_counter += 1
                if game_counter % 1e4 == 0:
                    logger.info(f"{game_counter} games processed")

                if game_counter >= args.max_games:
                    break
            except (ValueError, UnicodeDecodeError) as e:
                pass


if __name__ == '__main__':
    parse_pgn(parse_args())