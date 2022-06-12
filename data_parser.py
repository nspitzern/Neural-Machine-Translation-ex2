import os
from collections import Counter

import config_part1


def get_data(path, dataset_type):
    src_path = os.path.join(path, dataset_type + '.src')
    trg_path = os.path.join(path, dataset_type + '.trg')

    src_lines, trg_lines = _read_files(src_path, trg_path)

    return src_lines, trg_lines


def _read_files(src_path, trg_path):
    src_lines = []
    trg_lines = []

    # read both src and trg files
    for src_line, trg_line in zip(open(src_path, 'r', encoding='utf-8').readlines(),
                                  open(trg_path, 'r', encoding='utf-8').readlines()):
        src_lines.append([config_part1.START_TOKEN] + src_line.strip().split() + [config_part1.END_TOKEN])
        trg_lines.append([config_part1.START_TOKEN] + trg_line.strip().split() + [config_part1.END_TOKEN])

    return src_lines, trg_lines


def get_indices(lines):
    words_vocab = set()

    for line in lines:
        for word in line:
            words_vocab.add(word)

    # remove the predefined tokens, so they can be in the beginning of the mapping
    words_vocab.remove(config_part1.START_TOKEN)
    words_vocab.remove(config_part1.END_TOKEN)
    words_vocab.remove(config_part1.UNKNOWN_TOKEN)

    words_vocab = [config_part1.START_TOKEN, config_part1.END_TOKEN, config_part1.UNKNOWN_TOKEN, *sorted(words_vocab)]

    word2idx = {word: idx for idx, word in enumerate(words_vocab)}
    idx2word = {idx: word for idx, word in enumerate(words_vocab)}

    return word2idx, idx2word


def convert_rare_words(sentences, min_occurrence_threshold=config_part1.MIN_OCCURRENCES_THRESHOLD,
                       unknown_word=config_part1.UNKNOWN_TOKEN):
    c = Counter()
    words_to_convert = set()

    # count the occurrence of each word
    for line in sentences:
        c.update(line)

    # get a set of the words that appeared less than the min limit
    for word, count in c.items():
        if count <= min_occurrence_threshold:
            words_to_convert.add(word)

    # change all the words that needs conversion to unknown
    for i_line, line in enumerate(sentences):
        for i_word, word in enumerate(line):
            if word in words_to_convert:
                sentences[i_line][i_word] = unknown_word

    return sentences


def update_with_unknown(sentences, trg_word2idx, unknown_word):
    for i_line, line in enumerate(sentences):
        for i_word, word in enumerate(line):
            if word not in trg_word2idx:
                sentences[i_line][i_word] = unknown_word

    return sentences
