import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch

class BiLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.max_vocab_size,
            config.embed_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(2 * config.hidden_size, config.num_labels)

    def forward(self, x, seq_lengths):
        # x shape: (batch_size, seq_len)
        x_emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # 打包序列
        packed_input = pack_padded_sequence(
            x_emb,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM前向传播
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # 提取最后时刻的隐藏状态
        h_n = h_n.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        forward_last = h_n[-1, 0, :, :]  # (batch_size, hidden_size)
        backward_last = h_n[-1, 1, :, :]  # (batch_size, hidden_size)

        # 合并双向特征
        combined = torch.cat((forward_last, backward_last), dim=1)
        combined = self.dropout(combined)

        # 全连接层
        logits = self.fc(combined)
        return logits