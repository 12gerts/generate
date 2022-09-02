import re
import os
import sys
import numpy as np
import argparse
import pickle


def parse():
    """
    parsing command line arguments
    :return: Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir',
                        help='[OPTIONAL] enter the directory with the data. '
                             'if the parameter is skipped, press ENTER and enter the text')

    parser.add_argument('-m', '--model', required=True,
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
            print('wrong entry, try again')
            sys.exit()

        if len(re.sub(r'\s', '', str(text))) < 1:
            print('the line is empty. retry')
            sys.exit()

        parser.add_argument('--text', default=text)

    return parser.parse_args()


def tokenize(tokens):
    """
    splitting text into tokens
    """
    tokens = tokens.lower()
    tokens = re.sub(r'[^a-яё-]', ' ', str(tokens))
    tokens = re.sub(r'\s-\s', ' ', str(tokens))
    return tokens.split()


class Reader:
    def __init__(self, path_files=None, text=None):
        self.path = path_files
        if text is None:
            self.tokens = self.reading()
        else:
            self.tokens = tokenize(text)

    def reading(self):
        """
        read all files in a directory
        :return: the entire text
        """
        array_texts = []
        for file in os.listdir(self.path):
            with open(f'{self.path}\\{file}', encoding="utf8") as f:
                array_texts.append(f.read())
        sample_text = ' '.join(array_texts)
        return tokenize(sample_text)


def normalize(dictionary):
    """
    frequency normalization
    :param dictionary: original dictionary
    :return: modified dictionary
    """
    for key in dictionary:
        dictionary[key][0] = np.array(dictionary[key][0]) / np.array(dictionary[key][0]).sum()
    return dictionary


class Ngramm(Reader):
    def __init__(self, path_files=None, text=None):
        super().__init__(text=text, path_files=path_files)
        self.ngramm = self.create()

    def unigramm(self, ngramm, i):
        """
        add unigramm in a prefix dictionary
        """
        if self.tokens[i] in ngramm:
            possible_words = ngramm[self.tokens[i]]
            if self.tokens[i + 1] in possible_words:
                possible_words[0][possible_words.index(self.tokens[i + 1]) - 1] += 1
            else:
                possible_words.append(self.tokens[i + 1])
                possible_words[0].append(1)
            ngramm[self.tokens[i]] = possible_words
        else:
            ngramm[self.tokens[i]] = [[1], self.tokens[i + 1]]

    def bigramm(self, ngramm, i):
        """
        add bigramm in a prefix dictionary
        """
        if (self.tokens[i], self.tokens[i + 1]) in ngramm:
            possible_words = ngramm[(self.tokens[i], self.tokens[i + 1])]
            if self.tokens[i + 2] in possible_words:
                possible_words[0][possible_words.index(self.tokens[i + 2]) - 1] += 1
            else:
                possible_words.append(self.tokens[i + 2])
                possible_words[0].append(1)
            ngramm[(self.tokens[i], self.tokens[i + 1])] = possible_words
        else:
            ngramm[(self.tokens[i], self.tokens[i + 1])] = [[1], self.tokens[i + 2]]

    def create(self):
        """
        going through the text and compiling a prefix dictionary
        """
        ngramm = {}
        for i in range(len(self.tokens) - 1):
            try:
                self.unigramm(ngramm, i)
                self.bigramm(ngramm, i)
            except IndexError:
                break
        return normalize(ngramm)


def unloading(name: str, data: dict):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def main():
    parse_line = parse()
    if parse_line.input_dir is not None:
        prefix_dictionary = Ngramm(path_files=parse_line.input_dir)
    else:
        prefix_dictionary = Ngramm(text=parse_line.text)
    unloading(name=parse_line.model, data=prefix_dictionary.ngramm)


if __name__ == '__main__':
    main()
