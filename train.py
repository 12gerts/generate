import re
import os
import sys
import numpy as np
import argparse
import pickle


def parse():
    """ Parsing command line arguments. """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir',
                        help='[OPTIONAL] enter the directory with the data. '
                             'if the parameter is skipped, press ENTER and enter the text')

    parser.add_argument('-m', '--model',
                        required=True,
                        help='path to the file in which the model is saved')

    if parser.parse_args().input_dir is None:
        print('Does the text contain more than 1 line? y/n')
        agree = sys.stdin.readline()

        if agree[:-1].lower() == 'y':
            print('press ctrl+z on the new line to finish typing\ninput russian text:')
            text = sys.stdin.read()

        elif agree[:-1].lower() == 'n':
            print('input russian text:')
            text = sys.stdin.readline()

        else:
            print('[ERROR] wrong entry, try again')
            sys.exit()

        if len(re.sub(r'\s', '', str(text))) < 1:
            print('[ERROR] the line is empty. retry')
            sys.exit()

        parser.add_argument('--text', default=text)

    return parser.parse_args()


def tokenize(tokens):
    """ Splitting text into tokens. """
    tokens = tokens.lower()
    tokens = re.sub(r'[^a-яё-]', ' ', str(tokens))
    tokens = re.sub(r'\s-\s', ' ', str(tokens))
    return tokens.split()


def is_exist(path):
    """ Checking the existence of a file or directory. """
    if not os.path.exists(path):
        print('[ERROR] a directory or file that does not exist')
        sys.exit()


class TokenReader:
    """
    Reading files and processing the resulting text.
    Composing tokens.
    """
    def __init__(self, path_files=None, text=None):
        self.path = path_files
        self.text = text

    def reading(self):
        """ Read all files in a directory. """
        if self.text is not None:
            return tokenize(self.text)

        array_texts = []
        is_exist(self.path)

        for file in os.listdir(self.path):
            with open(f'{self.path}\\{file}', encoding="utf8") as f:
                array_texts.append(f.read())

        sample_text = ' '.join(array_texts)
        return tokenize(sample_text)


def normalize(dictionary):
    """ Frequency normalization. """
    for key in dictionary:
        occurrence_rate = np.array(dictionary[key][0])
        total_amount = np.array(dictionary[key][0]).sum()
        dictionary[key][0] = occurrence_rate / total_amount

    return dictionary


def ngramms(ngramm, key, value):
    """ Compiling N-gramm """
    if key in ngramm:
        possible_words = ngramm[key]

        if value in possible_words:
            word_index = possible_words.index(value) - 1
            possible_words[0][word_index] += 1
        else:
            possible_words.append(value)
            possible_words[0].append(1)

        ngramm[key] = possible_words
    else:
        ngramm[key] = [[1], value]


class Ngramm:
    """ Compiles a prefix dictionary from the list. """
    def __init__(self, tokens: list):
        self.tokens = tokens

    def __unigramm(self, ngramm: dict, i: int):
        """ Add unigramm in a prefix dictionary. """
        key = self.tokens[i]
        value = self.tokens[i + 1]
        return ngramms(ngramm, key, value)

    def __bigramm(self, ngramm: dict, i: int):
        """ Add bigramm in a prefix dictionary. """
        key = (self.tokens[i], self.tokens[i + 1])
        value = self.tokens[i + 2]
        return ngramms(ngramm, key, value)

    def fit(self):
        """ Going through the list and compiling a prefix dictionary. """
        ngramm = {}
        for i in range(len(self.tokens) - 2):
            self.__unigramm(ngramm, i)
            self.__bigramm(ngramm, i)
        self.__unigramm(ngramm, len(self.tokens) - 2)

        return normalize(ngramm)


def unloading(name: str, data: dict):
    """ Save dictionary to a file. """
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def main():
    parse_line = parse()
    if parse_line.input_dir is not None:
        token_reader = TokenReader(path_files=parse_line.input_dir)
    else:
        token_reader = TokenReader(text=parse_line.text)

    prefix_dictionary = Ngramm(token_reader.reading())

    unloading(name=parse_line.model,
              data=prefix_dictionary.fit())


if __name__ == '__main__':
    main()
