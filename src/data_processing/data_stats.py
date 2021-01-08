import numpy as np
from os import path
import glob
from prettytable import PrettyTable
import argparse
import re
from data_utils.chess_tokenizer import ChessTokenizer


def get_data_stats(data_dir, vocab_dir, notation, max_games=2e5):
    # Only LM testing related files
    data_dir = path.join(data_dir, notation)
    vocab_dir = path.join(vocab_dir, notation)

    tokenizer = ChessTokenizer(path.join(vocab_dir, "vocab.txt"), notation=notation)
    files = sorted([f for f in glob.glob(path.join(data_dir, '*.txt')) if re.match('.*/([a-z]*).txt', f)])
    assert (len(files) > 0)

    stats = PrettyTable(['Name', '# of games', 'Avg. game length', 'Max. game length',
                         'Avg. tokenized length', 'Max. tokenized length', 'Total moves'])
    for file in files:
        filename = path.basename(file).split(".")[0]
        token_len_list = []
        tokenized_len_list = []
        with open(file) as f:
            counter = 0
            for line in f:
                token_len_list.append(len(line.strip().split()))
                tokenized_len_list.append(len(tokenizer.encode(line.strip(), get_move_end_positions=False)))
                counter += 1
                if max_games is not None and counter >= max_games:
                    break

        token_len_list = np.array(token_len_list)
        stats.add_row([filename, len(token_len_list),
                       np.mean(token_len_list).round(decimals=1), np.max(token_len_list),
                       np.mean(tokenized_len_list).round(decimals=1), np.max(tokenized_len_list),
                       np.sum(token_len_list)])

    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--vocab_dir", type=str, help="Vocab directory", default="../models/vocab")
    parser.add_argument("--notation", default="uci", type=str, help="Notation")
    parser.add_argument("--max_games", default=2e5, type=int, help="Max games over which stats are calculated")

    args = parser.parse_args()
    get_data_stats(args.data_dir, args.vocab_dir, args.notation, max_games=args.max_games)

