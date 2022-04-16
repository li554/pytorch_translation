import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from transformer import Transformer
from datasets import Tokenizer, MyDataset, train_val_split


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
    # load model
    if opt.resume:
        start_epoch = 0
        # save_dir = 'runs/train/exp2'
        model = torch.load(opt.resume, map_location=device)
    else:
        start_epoch = 0
        model = Transformer(encoder_embedding_size, decoder_embedding_size, hidden_size, en_tokenizer.length(),
                            ch_tokenizer.length())
        model = model.to(device)
    t_total = len(train_iter) * opt.epochs
    warmup_steps = int(t_total * opt.warmup_proportion)

    def lr_lambda(step):
        # 线性变换，返回的是某个数值x，然后返回到类LambdaLR中，最终返回old_lr*x
        if step < warmup_steps:  # 增大学习率
            return float(step) / float(max(1, warmup_steps))
        # 减小学习率
        return max(0.0, float(t_total - step) / float(max(1.0, t_total - warmup_steps)))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': opt.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.epsilon)  # 用的是AdamW()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    writer = SummaryWriter(save_dir)
    cross_loss = nn.CrossEntropyLoss()
    cross_loss.to(device)

    best_acc = .0

    for e in range(start_epoch, epochs):
        model.train()
        for en_index, ch_index in tqdm(train_iter):
            en_index = en_index.to(device)
            ch_index = ch_index.to(device)
            pred = model(en_index, ch_index[:, :-1])
            label = ch_index[:, 1:].to(device)
            loss = cross_loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            optimizer.zero_grad()
            clip_grad_norm_(model.parameters(), 1)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss = loss.item()
        model.eval()
        val_acc, val_loss, n = .0, .0, 0
        for en_index, ch_index in val_iter:
            en_index = en_index.to(device)
            ch_index = ch_index.to(device)
            pred = model(en_index, ch_index[:, :-1])
            label = ch_index[:, 1:].to(device)
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
    parser.add_argument('--epochs', default=200, type=int)
    # parser.add_argument('--momentum', default=0.99, type=float)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_gamma', default=1, type=float, help='learning rate decrease rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup-proportion', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=int, default=1, help='grad clip strategy to train rnn')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--seed', default=42, type=int)
    # other config
    parser.add_argument('--batch_first', default=True, type=bool)
    parser.add_argument('--vocab', default='datas')
    parser.add_argument('--logdir', default='runs', type=str)
    parser.add_argument('--resume', default=False, type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--nums', default=None, type=int)
    opt = parser.parse_args()
    train(opt)
