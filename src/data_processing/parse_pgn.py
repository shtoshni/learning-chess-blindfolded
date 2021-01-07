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

    parser.add_argument("--source_dir", type=str, help="PGN dir")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--max_games", default=10_000_000, type=int, help="Max games to parse")

    parsed_args = parser.parse_args()
    assert (path.exists(parsed_args.source_dir))
    if not path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)

    return parsed_args


def parse_pgn(args):
    basename = path.basename(args.source_dir)  # path.splitext(path.basename(args.source_file))[0]
    output_file = path.join(args.output_dir, f"{basename}_uci.jsonl")
    files = glob.glob(path.join(args.source_dir, "fics*"))

    game_counter = 0
    with open(output_file, 'w') as output_f:
        for bz2_pgn_file in files:
            with bz2.open(bz2_pgn_file, 'rt') as pgn_f:
                while True:
                    try:
                        with suppress(ValueError):
                            game = pgn.read_game(pgn_f)
                            if game is None:
                                break

                        output_dict = OrderedDict()

                        headers = game.headers
                        key_attribs = ["White", "Black", "WhiteElo", "BlackElo", "Result"]
                        # key_attribs = ["White", "Black", "Result"]

                        for attrib in key_attribs:
                            if attrib in headers:
                                output_dict[attrib] = headers[attrib]
                            else:
                                output_dict[attrib] = None

                        game_moves = []
                        board = chess.Board()
                        for move in game.mainline_moves():
                            game_moves.append(board.uci(move))
                            board.push(move)

                        if len(game_moves) == 0:
                            continue
                        output_dict["moves"] = game_moves

                        output_f.write(json.dumps(output_dict) + "\n")

                        game_counter += 1
                        if game_counter % 1e4 == 0:
                            logger.info(f"{game_counter} games processed")
                            output_f.flush()

                        if game_counter >= args.max_games:
                            break
                    except (ValueError, UnicodeDecodeError, KeyError) as e:
                        pass


if __name__ == '__main__':
    parse_pgn(parse_args())
