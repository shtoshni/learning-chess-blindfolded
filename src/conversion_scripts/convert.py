import argparse
from os import path

from chess_utils.conversion_utils import convert_game_notation
from chess_utils.detection_utils import detect_move_notation


NOTATIONS = ['uci', 'lan_with_p', 'san', 'lan', 'rap_15', 'rap_85', 'rap_100']


def convert(args):
    with open(args.source_file) as source_f, open(args.output_file, 'w') as writer:
        for game_line in source_f:
            game = game_line.strip().split()
            conv_game = convert_game_notation(game, source_notation=args.source_notation,
                                              target_notation=args.target_notation)
            writer.write(" ".join(conv_game) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str, help="Input file that needs to be changed.")
    parser.add_argument("--output_file", type=str, help="Output file name")
    parser.add_argument("--source_notation", type=str, choices=NOTATIONS, help="Source notation")
    parser.add_argument("--target_notation", type=str, choices=NOTATIONS, help="Target notation")

    args = parser.parse_args()

    if args.source_notation is None:
        args.source_notation = detect_move_notation(args.source_file)
        print(f"Detected source notation: {args.source_notation}")

    if args.output_file is None:
        base_file = path.basename(args.source_file)
        if args.source_notation in base_file:
            output_base_file = base_file.replace(args.source_notation, args.target_notation)
        else:
            file_name, extension = base_file.split(".")
            output_base_file = file_name + "_" + args.target_notation + "." + extension

        args.output_file = path.join(path.dirname(args.source_file), output_base_file)
        print(f"Output file: {args.output_file}")

    return args


if __name__ == '__main__':
    args = parse_args()
    convert(args)