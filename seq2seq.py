import random

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_vocab_size, batch_first):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, encoder_embedding_num)
        self.lstm = nn.GRU(encoder_embedding_num, encoder_hidden_num, batch_first=batch_first)

    def forward(self, en_index):
        en_embedding = self.embedding(en_index)
        encoder_output, encoder_hidden = self.lstm(en_embedding)
        return encoder_output, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, ch_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)
        self.lstm = nn.GRU(decoder_embedding_num, decoder_hidden_num, batch_first=True)

    def forward(self, decoder_input, hidden):
        embedding = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)
        return decoder_output, decoder_hidden


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_state_t, encoder_outputs):
        b, s, h = encoder_outputs.shape
        attention_scores = torch.sum(
            torch.tile(decoder_state_t.unsqueeze(dim=1), dims=(s, 1)) * encoder_outputs, dim=-1)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        context = torch.sum(attention_scores.unsqueeze(dim=-1) * encoder_outputs, dim=1)
        return context, attention_scores


class AttentionDecoder(nn.Module):
    def __init__(self,
                 decoder_embedding_num, decoder_hidden_num, ch_vocab_size,
                 batch_first,
                 ch_tokenizer,
                 dropout=0.3,
                 teacher_force_prob=1.0,
                 teacher_force_gamma=0.99):
        super().__init__()
        self.embedding = nn.Embedding(ch_vocab_size, decoder_embedding_num)
        self.gru = nn.GRUCell(decoder_embedding_num, decoder_hidden_num)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.init_teacher_force_prob = teacher_force_prob
        self.teacher_force_prob = teacher_force_prob
        self.ch_tokenizer = ch_tokenizer
        self.teacher_force_gamma = teacher_force_gamma

    def forward(self, decoder_input, encoder_hidden, encoder_output, epoch):
        self.teacher_force_prob = self.init_teacher_force_prob * (self.teacher_force_gamma ** epoch)
        embed = self.embedding(decoder_input)
        if self.batch_first:
            b, s, h = embed.shape
        else:
            s, b, h = embed.shape
        ht = encoder_hidden[0]
        embed_pred = self.embedding(torch.tensor([[self.ch_tokenizer.BOS]], device=embed.device))
        embed_pred = torch.tile(embed_pred, dims=(b, 1, 1)) if self.batch_first else torch.tile(embed_pred,
                                                                                                dims=(1, b, 1))
        embed_pred = embed_pred[:, 0, :] if self.batch_first else embed_pred[0]
        if not self.batch_first:
            encoder_output = encoder_output.transpose(0, 1)
        decoder_output = []
        for t in range(s):
            embed_teacher = embed[:, t, :] if self.batch_first else embed[t]
            decoder_input = embed_teacher if random.random() < self.teacher_force_prob else embed_pred
            ht = self.gru(decoder_input, ht)
            context, _ = self.attention(ht, encoder_output)
            ht = self.dropout(ht)
            yt = torch.cat((ht, context), dim=-1)
            decoder_output.append(yt)
        decoder_output = torch.stack(decoder_output, dim=0)
        decoder_output = decoder_output.transpose(0, 1) if self.batch_first else decoder_output
        return decoder_output


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_embedding_num, encoder_hidden_num, en_corpus_len,
                 decoder_embedding_num, decoder_hidden_num, ch_corpus_len,
                 en_tokenizer, ch_tokenizer,
                 device='cpu',
                 batch_first=True,
                 dropout=0.3,
                 teacher_force_prob=1.0,
                 teacher_force_gamma=0.99):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num, encoder_hidden_num, en_corpus_len, batch_first)
        self.decoder = AttentionDecoder(decoder_embedding_num, decoder_hidden_num, ch_corpus_len, batch_first,
                                        ch_tokenizer, dropout, teacher_force_prob, teacher_force_gamma)
        self.projection = nn.Linear(2 * decoder_hidden_num, ch_corpus_len)
        self.device = device
        self.batch_first = batch_first
        self.en_tokenizer = en_tokenizer
        self.ch_tokenizer = ch_tokenizer

    def forward(self, en_index, ch_index, epoch):
        en_index = en_index.to(self.device)
        ch_index = ch_index.to(self.device)
        encoder_output, encoder_hidden = self.encoder(en_index)
        decoder_output = self.decoder(ch_index, encoder_hidden, encoder_output, epoch)
        return self.projection(decoder_output)

    def translate(self, sentence, en_tokenizer, ch_tokenizer, max_length=50):
        sentence = sentence.lower()
        with torch.no_grad():
            en_index = torch.tensor([en_tokenizer.encode(sentence)], device=self.device)
            if not self.batch_first:
                en_index = en_index.transpose(0, 1)
            encoder_output, encoder_hidden = self.encoder(en_index)
            decoder_input = torch.tensor([[ch_tokenizer.BOS]], device=self.device)
            ht = encoder_hidden[0]
            predictions = []
            for t in range(max_length):
                embed = self.decoder.embedding(decoder_input)
                embed = embed[:, 0, :] if self.batch_first else embed[0]
                ht = self.decoder.gru(embed, ht)
                context, _ = self.decoder.attention(ht, encoder_output)
                yt = torch.cat((ht, context), dim=-1)
                pred = self.projection(yt)
                w_index = int(torch.argmax(pred, dim=-1))
                word = ch_tokenizer.decode(w_index)
                if word == "<EOS>":
                    break
                predictions.append(word)
                decoder_input = torch.tensor([[w_index]], device=self.device)
            return "".join(predictions)
