import pickle
import numpy as np
import sys
import argparse
from train import tokenize, is_exist
import random


def loading(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def parse():
    """
    parsing command line arguments
    :return: Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model',
                        required=True,
                        help='path to the file from which the model is loaded')
    parser.add_argument('-l', '--length',
                        required=True,
                        type=int,
                        help=' the length of the generated sequence')
    parser.add_argument('-p', '--prefix',
                        nargs='+',
                        help='[OPTIONAL] the beginning of the sentence (one or more words)')

    return parser.parse_args()


def get_prefix(prefix):
    if prefix is None:
        return None
    if len(prefix) == 1:
        return prefix[0]
    return tuple(prefix[-2:])


class Generating:
    def __init__(self, ngramm=None, prefix=None, length=None):
        self.ngramm = ngramm
        self.full_prefix = prefix
        self.prefix = get_prefix(prefix)
        self.validate()
        self.length = length
        self.generated_string = self.generate()

    def random_key(self):
        return random.choice(list(self.ngramm))

    def get_value(self, current_prefix):
        return np.random.choice(
            self.ngramm[current_prefix][1:],
            p=self.ngramm[current_prefix][0]
        )

    def validate(self):
        try:
            self.ngramm[self.return_prefix()]
        except KeyError:
            print('incorrect input or insufficient data')
            sys.exit()

    def return_prefix(self):
        if self.prefix is None:
            return self.random_key()
        return self.prefix

    def generate(self):
        for _ in range(50):
            current_prefix = self.return_prefix()
            if isinstance(current_prefix, tuple):
                if self.full_prefix is None:
                    predictions_array = list(current_prefix)
                else:
                    predictions_array = self.full_prefix
            else:
                predictions_array = [current_prefix]

            try:
                while self.length > len(predictions_array):
                    prediction = self.get_value(current_prefix)
                    predictions_array.append(prediction)
                    current_prefix = get_prefix(predictions_array)
            except KeyError:
                continue

            return ' '.join(predictions_array)

        print('insufficient data')
        sys.exit()


def main():
    parse_line = parse()
    if parse_line.prefix is not None:
        prefix = tokenize(' '.join(parse_line.prefix))
    else:
        prefix = None

    is_exist(parse_line.model)
    print(
        Generating(
            ngramm=loading(parse_line.model),
            prefix=prefix,
            length=int(parse_line.length)
        ).generated_string
    )


if __name__ == '__main__':
    main()
