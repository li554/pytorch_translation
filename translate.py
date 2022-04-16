import argparse

import torch

from datasets import Tokenizer


def predict(opt):
    device = opt.device
    model = torch.load("runs/exp/weights/best.pt", map_location=device)
    en_tokenizer = Tokenizer(f"{opt.vocab}/en.vec", is_en=True)
    ch_tokenizer = Tokenizer(f"{opt.vocab}/ch.vec", is_en=False)
    while True:
        s = input("请输入英文:")
        s = model.translate(s, en_tokenizer, ch_tokenizer)
        print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--vocab', default='datas')
    opt = parser.parse_args()
    predict(opt)
