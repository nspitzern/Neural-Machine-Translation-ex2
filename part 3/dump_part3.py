import argparse

import torch
import sacrebleu

from data_parser import get_data, get_indices, convert_rare_words, update_with_unknown
from datasets import SentencesDataSet
from models import AttentionEncoder, AttentionDecoder
import config_part3
from utils import plot_graph, timeit, dump_heatmaps

DUMP_EXAMPLE_IDX = 0


def predict(decoder, trg, enc_output, trg_idx2word, device=config_part3.DEVICE):
    # make predictions
    prediction = [config_part3.START_TOKEN]

    dec_input = trg[0]

    dec_hidden = decoder.init_hidden(device)
    # create initializing context
    dec_context = torch.zeros(1, 1, config_part3.HIDDEN_SIZE, device=device)

    for i in range(config_part3.MAX_SENTENCE_LENGTH):
        dec_output, dec_hidden, dec_context, dec_attention = decoder(dec_input, enc_output, dec_hidden, dec_context)

        # get index of the next word
        dec_input = dec_output.argmax()

        # convert to word
        prediction.append(trg_idx2word[dec_input.item()])

        # check if sentence reached the length limit
        if trg_idx2word[dec_input.item()] == config_part3.END_TOKEN:
            break

    prediction = ' '.join(prediction)

    return prediction


def evaluate(epoch, dev_data, encoder, decoder, criterion, trg_idx2word, dev_src_text, dev_trg_text,
             device=config_part3.DEVICE):
    encoder.eval()
    decoder.eval()

    # init loss for the evaluation
    dev_loss = 0
    predictions = []

    with torch.no_grad():
        # go over the dataset
        for idx, (src, trg) in enumerate(dev_data):
            attention_maps = []
            target_length = trg.size(0)

            # init sentence loss
            sentence_loss = 0

            # pass through the encoder
            enc_output, enc_hidden = encoder(src)

            dec_hidden = decoder.init_hidden(device)
            # create initializing context
            dec_context = torch.zeros(1, 1, config_part3.HIDDEN_SIZE, device=device)

            # pass through the decoder
            for i in range(target_length - 1):
                dec_output, dec_hidden, dec_context, dec_attention = decoder(trg[i], enc_output, dec_hidden,
                                                                             dec_context)
                sentence_loss += criterion(dec_output, trg[i + 1])

                attention_maps.append(dec_attention)

            dev_loss += (sentence_loss.item() / (target_length - 1))

            predicted_words = predict(decoder, trg, enc_output, trg_idx2word, device)

            predictions.append(predicted_words)

            # dump heatmap of attention maps for the first sentence in the dev dataset
            if idx == DUMP_EXAMPLE_IDX:
                dump_heatmaps(epoch, dev_src_text[idx][1:len(dev_src_text[idx]) - 1],
                              dev_trg_text[idx][1:len(dev_trg_text[idx]) - 1],
                              torch.stack(attention_maps))

        refs = [' '.join(ref) for ref in dev_trg_text]

        # get BLEU score for translations
        bleu = sacrebleu.corpus_bleu(predictions, [refs])

    return dev_loss / len(dev_data), bleu.score


@timeit
def train(train_data, dev_data, encoder, decoder, trg_idx2word, dev_src_text, dev_trg_text, epochs, device):
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # init criterion
    criterion = config_part3.CRITERION()

    # init optimizers
    enc_opt = config_part3.OPTIMIZER(encoder.parameters(), lr=config_part3.LR)
    dec_opt = config_part3.OPTIMIZER(decoder.parameters(), lr=config_part3.LR)

    # init losses and BLEU scores
    dev_losses = []
    bleu_scores = []
    best_bleu = 0

    # Start epochs
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()

        # init total loss for current epoch
        total_loss = 0

        # Go over the dataset
        for src, trg in train_data:
            # init gradients
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            target_length = trg.size(0)

            # init loss for current sentence
            sentence_loss = 0

            # pass through the encoder
            enc_output, enc_hidden = encoder(src)

            dec_hidden = decoder.init_hidden(device)

            # create initializing context
            dec_context = torch.zeros(1, 1, config_part3.HIDDEN_SIZE, device=device)

            # pass through the decoder
            for i in range(target_length - 1):
                dec_output, dec_hidden, dec_context, dec_attention = decoder(trg[i], enc_output, dec_hidden,
                                                                             dec_context)

                sentence_loss += criterion(dec_output, trg[i + 1])

            # scale sentence loss by the length of the sentence
            total_loss += (sentence_loss.item() / (target_length - 1))

            # calculate grads
            sentence_loss.backward()

            enc_opt.step()
            dec_opt.step()

        # scale loss by the size of the dataset
        train_loss = total_loss / len(train_data)

        # evaluate model
        dev_loss, bleu_score = evaluate(epoch, dev_data, encoder, decoder, criterion, trg_idx2word, dev_src_text,
                                        dev_trg_text, device)

        dev_losses.append(dev_loss)
        bleu_scores.append(bleu_score)

        print(
            f'[{epoch}/{config_part3.EPOCHS}] Training loss: {train_loss}, Validation loss: {dev_loss}, BLEU score: {bleu_score}')

        if bleu_score >= best_bleu:
            print(f'New best BLEU score! Previous: {best_bleu}, New: {bleu_score}')
            best_bleu = bleu_score

    return dev_losses, bleu_scores


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse the data for the train set
    train_src_lines, train_trg_lines = get_data(args.path, 'train')
    train_src_lines = convert_rare_words(train_src_lines,
                                         min_occurrence_threshold=config_part3.MIN_OCCURRENCES_THRESHOLD)
    train_trg_lines = convert_rare_words(train_trg_lines,
                                         min_occurrence_threshold=config_part3.MIN_OCCURRENCES_THRESHOLD)

    # convert each word into its corresponding index
    src_word2idx, src_idx2word = get_indices(train_src_lines)
    trg_word2idx, trg_idx2word = get_indices(train_trg_lines)

    # parse the data for the dev set
    dev_src_lines, dev_trg_lines = get_data(args.path, 'dev')
    dev_trg_lines = update_with_unknown(dev_trg_lines, trg_word2idx, unknown_word=config_part3.UNKNOWN_TOKEN)

    dev_src_text = dev_src_lines.copy()
    dev_trg_text = dev_trg_lines.copy()

    # create datasets
    train_dataset = SentencesDataSet(train_src_lines, train_trg_lines, src_word2idx, trg_word2idx, shuffle=True,
                                     device=device)
    dev_dataset = SentencesDataSet(dev_src_lines, dev_trg_lines, src_word2idx, trg_word2idx, device=device)

    # create encoder model
    encoder = AttentionEncoder(input_size=len(src_word2idx),
                               embedding_size=config_part3.EMBEDDING_SIZE,
                               hidden_size=config_part3.HIDDEN_SIZE,
                               dropout=config_part3.DROP_RATE)

    # create decoder model
    decoder = AttentionDecoder(output_size=len(trg_word2idx),
                               embedding_size=config_part3.EMBEDDING_SIZE,
                               hidden_size=config_part3.HIDDEN_SIZE,
                               dropout=config_part3.DROP_RATE)

    # start training
    losses, bleu_scores = train(train_dataset,
                                dev_dataset,
                                encoder,
                                decoder,
                                trg_idx2word,
                                dev_src_text,
                                dev_trg_text,
                                epochs=config_part3.EPOCHS,
                                device=device)

    plot_graph(losses, "Dev Loss")
    plot_graph(bleu_scores, "BLEU score")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True, help='Path to train and dev files')

    args = parser.parse_args()
    main(args)
