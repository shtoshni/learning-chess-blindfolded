from os import path
import argparse
import random
import chess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str, help="File with all games.")
    parser.add_argument("--min_moves", default=10, type=int, help="Minimum number of moves in games.")
    parser.add_argument("--max_moves", default=150, type=int, help="Maximum number of moves in games.")
    parser.add_argument("--output_file", type=str, help="Output file.")

    args = parser.parse_args()
    assert(path.exists(args.source_file))

    if args.output_file is None:
        basename, extension = path.splitext(path.basename(args.source_file))
        args.output_file = path.join(path.dirname(args.source_file), basename + "-uniq" + extension)

    return args


def filter_data(args):
    """Filter data with following objectives:
    (1) Unique games
    (2) MIN moves <= Game length <= MAX moves
    (3) Games lacking exact moves, moves annotated as -- or 0000
    """
    data_hash = set()

    num_games = 0
    num_filtered_games = 0

    games = []
    with open(args.source_file) as reader:
        for line in reader:
            game = line.strip()
            num_games += 1
            if '0000' in game or "--" in game:
                # Error in game, missing information
                continue
            num_moves = len(game.split())
            if num_moves < args.min_moves or num_moves > args.max_moves:
                continue
            game_hash = hash(game)
            if game_hash in data_hash:
                # Game seen
                continue

            data_hash.add(game_hash)
            games.append(line)
            num_filtered_games += 1

    with open(args.output_file, 'w') as writer:
        random.seed(10)
        random.shuffle(games)

        for game in games:
            try:
                # Sanity check for filtered games
                moves = game.strip().split()
                board = chess.Board()
                for move in moves:
                    board.push(board.parse_uci(move))
                writer.write(game)
            except ValueError:
                pass

    print(f"# of games: {num_games}, # of filtered games: {num_filtered_games}")
    print(f"Filtered games at {args.output_file}")


if __name__ == '__main__':
    filter_data(parse_args())

