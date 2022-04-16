import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from datasets import Tokenizer, MyDataset, train_val_split
from seq2seq import Seq2Seq


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def train(opt):
    set_seed(opt.seed)
    batch_size, encoder_embedding_size, decoder_embedding_size, hidden_size, lr, epochs, device, logdir, \
    batch_first, num_workers = \
        opt.batch_size, opt.encoder_embedding_size, opt.decoder_embedding_size, \
        opt.hidden_size, opt.lr, opt.epochs, opt.device, opt.logdir, opt.batch_first, opt.num_workers

    en_tokenizer = Tokenizer(f"{opt.vocab}/en.vec", is_en=True)
    ch_tokenizer = Tokenizer(f"{opt.vocab}/ch.vec", is_en=False)
    dataset = MyDataset(f"{opt.vocab}/translate.csv", en_tokenizer, ch_tokenizer, nums=opt.nums,
                        batch_first=batch_first)
    train_iter, val_iter = train_val_split(dataset, batch_size, num_workers)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    temp = len(os.listdir(logdir))
    save_dir = os.path.join(logdir, 'exp' + ('' if temp == 0 else str(temp)))

    # 模型
    model = Seq2Seq(encoder_embedding_size, hidden_size, en_tokenizer.length(), decoder_embedding_size, hidden_size,
                    ch_tokenizer.length(), en_tokenizer, ch_tokenizer, device=device, batch_first=batch_first,
                    dropout=0.1,
                    teacher_force_prob=opt.tp_prob,
                    teacher_force_gamma=opt.tp_gamma)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_gamma)

    # 损失函数
    cross_loss = nn.CrossEntropyLoss()
    cross_loss.to(device)

    # 绘制
    writer = SummaryWriter(save_dir)

    best_acc = .0

    for e in range(epochs):
        model.train()
        for en_index, ch_index in tqdm(train_iter):
            en_index = en_index.to(device)
            ch_index = ch_index.to(device)
            pred = model(en_index, ch_index[:, :-1] if batch_first else ch_index[-1], e)
            label = (ch_index[:, 1:] if batch_first else ch_index[1:]).to(device)
            loss = cross_loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            optimizer.zero_grad()
            clip_grad_norm_(model.parameters(), opt.grad_clip)
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_loss = loss.item()
        model.eval()
        val_acc, val_loss, n = .0, .0, 0
        for en_index, ch_index in val_iter:
            en_index = en_index.to(device)
            ch_index = ch_index.to(device)
            pred = model(en_index, ch_index[:, :-1] if batch_first else ch_index[-1], e)
            label = (ch_index[:, 1:] if batch_first else ch_index[1:]).to(device)
            loss = cross_loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            val_acc += torch.sum(pred.argmax(dim=-1) == label)
            val_loss += loss.item()
            n += label.shape[0] * label.shape[1]
        val_acc /= n
        val_loss /= n
        writer.add_scalar('x/lr', optimizer.state_dict()['param_groups'][0]['lr'], e)
        writer.add_scalar("train/loss", train_loss, e)
        writer.add_scalar("val/loss", val_loss, e)
        writer.add_scalar("val/acc", val_acc, e)
        print(f"epoch {e}  train loss {loss.item()}  val loss {val_loss}  val acc {val_acc}")

        # 保存模型
        if val_acc > best_acc:
            if not os.path.exists(os.path.join(save_dir, 'weights')):
                os.makedirs(os.path.join(save_dir, 'weights'))
            torch.save(model, os.path.join(save_dir, "weights/best.pt"))
        torch.save(model, os.path.join(save_dir, "weights/last.pt"))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")
    # network structure
    parser.add_argument('--encoder_embedding_size', default=128, type=int)
    parser.add_argument('--decoder_embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    # hyper-parameters
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_gamma', default=0.99, type=float, help='learning rate decrease rate')
    parser.add_argument('--tp_prob', default=1, type=float, help='learning rate decrease rate')
    parser.add_argument('--tp_gamma', default=1, type=float, help='learning rate decrease rate')
    # parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=int, default=1, help='grad clip strategy to train rnn')
    # parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--seed', default=42, type=int)
    # other config
    parser.add_argument('--batch_first', default=True, type=bool)
    parser.add_argument('--vocab', default='datas')
    parser.add_argument('--logdir', default='runs', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--nums', default=None, type=int)
    opt = parser.parse_args()
    train(opt)
