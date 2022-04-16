import pickle
import re

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, SequentialSampler, DataLoader


class MyDataset(Dataset):
    @staticmethod
    def get_data(dataset_path, nums=None):
        all_datas = pd.read_csv(dataset_path)
        en_data = list(all_datas["english"])
        ch_data = list(all_datas["chinese"])
        if nums is None:
            return en_data, ch_data
        else:
            return en_data[:nums], ch_data[:nums]

    def __init__(self, dataset_path, en_tokenizer, ch_tokenizer, nums=None, batch_first=True):
        en_data, ch_data = self.get_data(dataset_path, nums=nums)
        self.en_data = en_data
        self.ch_data = ch_data
        self.en_tokenizer = en_tokenizer
        self.ch_tokenizer = ch_tokenizer
        self.batch_first = batch_first

    def __getitem__(self, index):
        en = self.en_data[index]
        en = en.lower()
        ch = self.ch_data[index]
        en_index = self.en_tokenizer.encode(en)
        ch_index = self.ch_tokenizer.encode(ch)
        return en_index, ch_index

    def collate_fn(self, batch_list):
        en_index, ch_index = [], []
        for en, ch in batch_list:
            en_index.append(torch.tensor(en))
            ch_index.append(torch.tensor([self.ch_tokenizer.BOS] + ch + [self.ch_tokenizer.EOS]))

        en_index = pad_sequence(en_index, batch_first=True, padding_value=self.en_tokenizer.PAD)
        ch_index = pad_sequence(ch_index, batch_first=True, padding_value=self.ch_tokenizer.PAD)

        if not self.batch_first:
            en_index = en_index.transpose(0, 1)
            ch_index = ch_index.transpose(0, 1)
        return en_index, ch_index

    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)


class RegexpReplacer(object):
    def __init__(self, patterns=None):
        if patterns is None:
            patterns = [
                (r'won\'t', 'will not'),
                (r'can\'t', 'cannot'),
                (r'i\'m', 'i am'),
                (r'ain\'t', 'is not'),
                (r'(\w+)\'ll', '\g<1> will'),
                (r'(\w+)n\'t', '\g<1> not'),
                (r'(\w+)\'ve', '\g<1> have'),
                (r'(\w+)\'s', '\g<1> is'),
                (r'(\w+)\'re', '\g<1> are'),
                (r'(\w+)\'d', '\g<1> would')
            ]
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


class Tokenizer:
    def __init__(self, vocab_path, is_en=True):
        with open(vocab_path, "rb") as f1:
            _, word2index, index2word = pickle.load(f1)
            f1.close()
        if is_en:
            word2index = {word: index + 1 for word, index in word2index.items()}
            word2index.update({"<PAD>": 0})
            index2word = ["<PAD>"] + index2word
        else:
            word2index = {word: index + 3 for word, index in word2index.items()}
            word2index.update({"<PAD>": 0, "<BOS>": 1, "<EOS>": 2})
            index2word = ["<PAD>", "<BOS>", "<EOS>"] + index2word
        self.word2index = word2index
        self.index2word = index2word
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2

    def encode(self, sentence):
        return [self.word2index[w] for w in sentence]

    def decode(self, index):
        return self.index2word[index]

    def length(self):
        return len(self.index2word)


def train_val_split(dataset, batch_size, num_workers, validation_split=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SequentialSampler(train_indices)
    valid_sampler = SequentialSampler(val_indices)
    train_iter = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    valid_iter = DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    return train_iter, valid_iter


if __name__ == '__main__':
    tokenizer = Tokenizer(vocab_path="datas/en.vec", is_en=True)
    print(tokenizer.encode("I'm do that."))
