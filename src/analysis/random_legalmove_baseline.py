import glob
import json
import argparse
from os import path

from data_utils.utils import get_eval_set_name
from constants import TASK_CATEGORIES


def get_category_accuracy(eval_files, legal_category="end"):
    for eval_file in eval_files:
        category = get_eval_set_name(eval_file)
        results = []
        with open(eval_file) as f:
            for line in f.readlines():
                instance = json.loads(line.strip())
                results.append(1 / len(instance[f"legal_{legal_category}ing_square"]))

        assert (len(results) == 1000)
        print(f'{category}: {sum(results) * 100 / len(results): .1f}')


def get_random_accuracy(data_dir):
    other_eval_dir = path.join(data_dir, "other_eval/uci")
    for task_category in TASK_CATEGORIES:
        eval_files = glob.glob(path.join(other_eval_dir, f"*{task_category}*.jsonl"))
        get_category_accuracy(eval_files, legal_category=task_category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="Root directory where other evaluation files are present")
    args = parser.parse_args()

    get_random_accuracy(args.data_dir)


