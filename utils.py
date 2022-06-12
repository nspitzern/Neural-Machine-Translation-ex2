from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def timeit(func):
    def timed(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()

        print(f'Execution time of function "{func.__name__}" elapsed {(end - start)} seconds')

        return result
    return timed


def plot_graph(data, label=''):
    x = np.arange(len(data))

    plt.plot(x, data)

    plt.ylabel(label)
    plt.xlabel('Epochs')

    plt.xticks(np.arange(0, len(data)))

    plt.show()


def dump_heatmaps(epoch, original_sentence, translation_sentence, attention_maps):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention_maps = attention_maps.squeeze().cpu().detach().numpy()

    ax.set_xticks(np.arange(len(original_sentence) + 2))
    ax.set_yticks(np.arange(len(original_sentence) + 1))

    ax = sns.heatmap(attention_maps, square=True, cmap='Blues',
                     xticklabels=['begin'] + original_sentence + ['end'],
                     annot=True, cbar=False)

    ax.set_yticklabels(translation_sentence + ['end'], rotation=360)

    ax.set(title=f'Epoch_{epoch}\n{" ".join(original_sentence)} -> {" ".join(translation_sentence)}')

    fig.tight_layout()
    plt.savefig(f"Epoch_{epoch}_dump")
    plt.close()
