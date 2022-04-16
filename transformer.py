import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        max_len = 5000
        pe = torch.zeros(size=(max_len, self.input_size))
        pos = torch.arange(max_len)
        i2s = torch.arange(self.input_size) // 2 * 2
        item = pos.unsqueeze(1) * torch.exp(i2s * self.input_size * math.log(10000))
        pe[:, 0::2] = torch.sin(item)[:, 0::2]
        pe[:, 1::2] = torch.cos(item)[:, 1::2]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_embedding):
        x = input_embedding + self.pe[:, :input_embedding.shape[1]]
        x = self.dropout(x)
        return x


def generate_attention_mask(encoder_input, decoder_input):
    sd = decoder_input.shape[1]
    attention_mask = torch.tile(encoder_input.eq(0).unsqueeze(dim=1), dims=(1, sd, 1))
    return attention_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dk = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0
        self.Q = nn.Linear(embedding_size, hidden_size)
        self.K = nn.Linear(embedding_size, hidden_size)
        self.V = nn.Linear(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.layernorm = nn.LayerNorm(embedding_size)

    def forward(self, embedding, encoder_output, attention_mask):
        residual = embedding
        q = self.Q(embedding)
        k = self.K(encoder_output)
        v = self.V(encoder_output)
        b, s, h = q.shape
        q = q.view(b, -1, self.num_heads, self.dk).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.dk).transpose(1, 2)
        attention_score = torch.matmul(q, k.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.dk)
        if attention_mask is not None:
            attention_score = attention_score.masked_fill_(
                torch.tile(attention_mask.unsqueeze(1), dims=(1, self.num_heads, 1, 1)), float("-inf"))
        attention_score = torch.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        context = torch.matmul(attention_score, v)
        context = context.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.dense(context) + residual
        output = self.layernorm(output)
        return output, attention_score


# (b,s,e)
class PoswiseFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.layernorm = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = x.transpose(-1, -2)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = y.transpose(-1, -2)
        y = self.layernorm(y)
        return x + y


class EncoderLayer(nn.Module):
    def __init__(self, encoder_embedding_num, hidden_size, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(encoder_embedding_num, hidden_size, num_heads)
        self.fn = PoswiseFeedForward(encoder_embedding_num, hidden_size)

    def forward(self, encoder_embedding, attention_mask):
        x, attention_states = self.self_attention(encoder_embedding, encoder_embedding, attention_mask)
        x = self.fn(x)
        return x, attention_states


class Encoder(nn.Module):
    def __init__(self, encoder_embedding_num, hidden_size, en_vocab_size, n_layers=6, num_heads=8):
        super().__init__()
        self.input_embedding = nn.Embedding(en_vocab_size, encoder_embedding_num)
        self.positional_encoding = PositionalEncoding(encoder_embedding_num)
        self.layers = nn.ModuleList(
            [EncoderLayer(encoder_embedding_num, hidden_size, num_heads) for _ in range(n_layers)])

    def forward(self, encoder_input):
        x = self.input_embedding(encoder_input)
        x = self.positional_encoding(x)
        attention_mask = generate_attention_mask(encoder_input, encoder_input)
        attention_states = []
        for layer in self.layers:
            x, attention_state = layer(x, attention_mask)
            attention_states.append(attention_state)
        return x, attention_states


class DecoderLayer(nn.Module):
    def __init__(self, decoder_embedding_num, hidden_size, num_heads):
        super().__init__()
        self.mask_attention = MultiHeadAttention(decoder_embedding_num, hidden_size, num_heads)
        self.inter_attention = MultiHeadAttention(decoder_embedding_num, hidden_size, num_heads)
        self.fn = PoswiseFeedForward(decoder_embedding_num, hidden_size)

    def forward(self, decoder_embedding, encoder_output, mask_attention_mask, inter_attention_mask):
        x, _ = self.mask_attention(decoder_embedding, decoder_embedding, mask_attention_mask)
        x, _ = self.inter_attention(x, encoder_output, inter_attention_mask)
        x = self.fn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_embedding_num, hidden_size, ch_vocab_size, n_layers=6, num_heads=8):
        super().__init__()
        self.output_embedding = nn.Embedding(ch_vocab_size, decoder_embedding_num)
        self.position_encoding = PositionalEncoding(decoder_embedding_num)
        self.layers = nn.ModuleList(
            [DecoderLayer(decoder_embedding_num, hidden_size, num_heads) for _ in range(n_layers)])

    def forward(self, decoder_input, encoder_input, encoder_output):
        mask_attention_mask_1 = generate_attention_mask(decoder_input, decoder_input)
        mask_attention_mask_2 = torch.triu(
            torch.ones(decoder_input.shape[1], decoder_input.shape[1], device=decoder_input.device), diagonal=1)
        mask_attention_mask = (mask_attention_mask_1 + mask_attention_mask_2).gt(0)
        inter_attention_mask = generate_attention_mask(encoder_input, decoder_input)

        decoder_embedding = self.output_embedding(decoder_input)
        decoder_embedding = self.position_encoding(decoder_embedding)

        x = decoder_embedding
        for layer in self.layers:
            x = layer(x, encoder_output, mask_attention_mask, inter_attention_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, encoder_embedding_num, decoder_embedding_num, hidden_size, en_vocab_size, ch_vocab_size,
                 num_layers=6,
                 num_heads=8):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num, hidden_size, en_vocab_size, num_layers, num_heads)
        self.decoder = Decoder(decoder_embedding_num, hidden_size, ch_vocab_size, num_layers, num_heads)
        self.classifier = nn.Linear(decoder_embedding_num, ch_vocab_size)

    def forward(self, encoder_input, decoder_input):
        encoder_output, attention_states = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_input, encoder_output)
        logits = self.classifier(decoder_output)
        return logits

    def translate(self, sentence, en_tokenizer, ch_tokenizer, max_length=50):
        with torch.no_grad():
            encoder_input = torch.tensor([en_tokenizer.encode(sentence)], device=self.device)
            encoder_output, attention_states = self.encoder(encoder_input)
            decoder_input = torch.tensor([[ch_tokenizer.BOS]], device=self.device)
            result = ""
            for i in range(max_length):
                decoder_output = self.decoder(decoder_input, encoder_input, encoder_output)
                logits = self.classifier(decoder_output)
                windex = logits.argmax(dim=-1)
                if windex[0][-1] == 2:
                    break
                decoder_input = torch.cat((decoder_input, windex[:, -1].unsqueeze(0)), dim=1)
                result += ch_tokenizer.decode(windex[0][-1])
            return result
