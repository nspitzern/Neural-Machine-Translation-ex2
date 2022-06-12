import torch.optim as optim
import torch.nn as nn

EPOCHS = 10
LR = 3e-4
OPTIMIZER = optim.Adam
CRITERION = nn.NLLLoss
BATCH_SIZE = 1
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
DROP_RATE = 0.3

DEVICE = 'cpu'

MAX_SENTENCE_LENGTH = 15
START_TOKEN = '<s>'
END_TOKEN = '</s>'
UNKNOWN_TOKEN = '<unk>'
MIN_OCCURRENCES_THRESHOLD = 3

MAPS_PATH = 'maps_part2.pth'
MODEL_CHECKPOINT_PATH = 'models_part2_checkpoint.pth'
