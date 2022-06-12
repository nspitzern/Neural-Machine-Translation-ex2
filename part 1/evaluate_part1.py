import argparse

import torch

from data_parser import get_data, update_with_unknown
from datasets import SentencesDataSet
from models import Encoder, Decoder
import config_part1
from train_part1 import evaluate
from utils import timeit


@timeit
def do_evaluation(*args, **kwargs):
    return evaluate(*args, **kwargs)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maps = torch.load(config_part1.MAPS_PATH)

    src_word2idx = maps['src_word2idx']
    trg_word2idx = maps['trg_word2idx']
    src_idx2word = maps['src_idx2word']
    trg_idx2word = maps['trg_idx2word']

    # parse the data for the dev set
    test_src_lines, test_trg_lines = get_data(args.path, 'test')
    test_trg_lines = update_with_unknown(test_trg_lines, trg_word2idx, unknown_word=config_part1.UNKNOWN_TOKEN)

    test_trg_text = test_trg_lines.copy()

    test_dataset = SentencesDataSet(test_src_lines, test_trg_lines, src_word2idx, trg_word2idx, device=device)

    models = torch.load(config_part1.MODEL_CHECKPOINT_PATH)

    criterion = config_part1.CRITERION()

    # create encoder model
    encoder = Encoder(input_size=len(src_word2idx),
                      embedding_size=config_part1.EMBEDDING_SIZE,
                      hidden_size=config_part1.HIDDEN_SIZE,
                      dropout=config_part1.DROP_RATE)

    # create decoder model
    decoder = Decoder(output_size=len(trg_word2idx),
                      embedding_size=config_part1.EMBEDDING_SIZE,
                      hidden_size=config_part1.HIDDEN_SIZE,
                      dropout=config_part1.DROP_RATE)

    encoder.load_state_dict(models['encoder'])
    decoder.load_state_dict(models['decoder'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # start training
    test_loss, bleu_score = do_evaluation(test_dataset,
                                          encoder,
                                          decoder,
                                          criterion,
                                          trg_idx2word,
                                          test_trg_text,
                                          device=device)

    print(f'Test Loss: {test_loss}, Test BLEU: {bleu_score}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-p', '--path', required=True, type=str, help='The path to the data')

    args = args_parser.parse_args()
    main(args)
