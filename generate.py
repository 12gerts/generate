import pickle
import numpy as np
import sys
import argparse
from train import tokenize, is_exist
import random


def loading(file_name):
    """ Loads data from the model. """
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def parse():
    """ Parsing command line arguments. """
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
    """ Returns the desired prefix format. """
    if prefix is None:
        return None
    if len(prefix) == 1:
        return prefix[0]
    return tuple(prefix[-2:])


def error_handling(error):
    """ Error output. """
    if error == 'prefix_not_found':
        print('[ERROR] incorrect input or insufficient data. Prefix not found')
    elif error == 'generate_error':
        print('[ERROR] insufficient data. The sequence cannot be generated with the given prefix\\length')
    sys.exit()


class Generating:
    def __init__(self, ngramm=None, prefix=None, length=None):
        self.ngramm = ngramm
        self.full_prefix = prefix
        self.prefix = get_prefix(prefix)
        self.__validate()
        self.length = length

    def generate(self):
        """ Sequence generation. """
        for _ in range(50):
            current_prefix = self.__return_prefix()

            if isinstance(current_prefix, tuple):
                if self.full_prefix is None:
                    predictions_array = list(current_prefix)
                else:
                    predictions_array = self.full_prefix
            else:
                predictions_array = [current_prefix]

            try:
                while self.length > len(predictions_array):
                    prediction = self.__get_value(current_prefix)
                    predictions_array.append(prediction)
                    current_prefix = get_prefix(predictions_array)
            except KeyError:
                continue

            return ' '.join(predictions_array)

        error_handling('generate_error')

    def __random_key(self):
        """ Return a random key. """
        return random.choice(list(self.ngramm))

    def __get_value(self, current_prefix):
        """ Choosing a possible continuation of a sentence. """
        return np.random.choice(
            self.ngramm[current_prefix][1:],
            p=self.ngramm[current_prefix][0]
        )

    def __validate(self):
        """ Checking for the existence of a prefix. """
        if self.__return_prefix() not in self.ngramm:
            error_handling('prefix_not_found')

    def __return_prefix(self):
        """ Return prefix or random prefix. """
        if self.prefix is None:
            return self.__random_key()
        return self.prefix


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
        ).generate()
    )


if __name__ == '__main__':
    main()
