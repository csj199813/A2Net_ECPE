import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = 'data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'


# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = 'src/bert-base-chinese'
        self.epochs = 50
        self.lr = 2e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 2


