import os
import json
import re
import jieba
from collections import Counter
import torch
from torch.utils.data import Dataset


def clean_and_tokenize(text):
    """清洗文本并分词"""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = re.sub(r'\d+', '[NUM]', text)
    return jieba.lcut(text)


def build_vocab(texts, max_vocab_size):
    """构建词汇表"""
    special_tokens = ['[PAD]', '[UNK]', '[NUM]']
    counter = Counter()
    for words in texts:
        counter.update(words)

    # 保留高频词
    vocab = special_tokens + [word for word, _ in counter.most_common(max_vocab_size - len(special_tokens))]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx


def load_data(data_path, max_vocab_size, max_seq_len):
    """加载并预处理数据"""
    label_mapping = {
        "100": 0, "101": 1, "102": 2, "103": 3, "104": 4,
        "106": 5, "107": 6, "108": 7, "109": 8, "110": 9,
        "112": 10, "113": 11, "115": 12, "116": 13, "114": 14
    }

    # 加载训练数据
    train_texts = []
    train_labels = []
    train_path = os.path.join(data_path, 'train.json')
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            words = clean_and_tokenize(item['sentence'])
            train_texts.append(words)
            train_labels.append(label_mapping[item['label']])

    # 构建词汇表
    word_to_idx = build_vocab(train_texts, max_vocab_size)

    # 加载验证数据
    val_texts = []
    val_labels = []
    dev_path = os.path.join(data_path, 'dev.json')
    if os.path.exists(dev_path):
        with open(dev_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                words = clean_and_tokenize(item['sentence'])
                val_texts.append(words)
                val_labels.append(label_mapping[item['label']])

    return train_texts, val_texts, train_labels, val_labels, word_to_idx


class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_seq_len = max_seq_len
        self.unk_idx = word_to_idx.get('[UNK]', 1)
        self.pad_idx = word_to_idx.get('[PAD]', 0)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx]
        label = self.labels[idx]

        # 转换词序列为索引
        indices = [self.word_to_idx.get(word, self.unk_idx) for word in words]
        seq_len = len(indices)

        # 截断或填充
        if seq_len > self.max_seq_len:
            indices = indices[:self.max_seq_len]
            seq_len = self.max_seq_len
        else:
            indices += [self.pad_idx] * (self.max_seq_len - seq_len)

        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'seq_lengths': seq_len,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    seq_lengths = torch.tensor([item['seq_lengths'] for item in batch], dtype=torch.long)
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'seq_lengths': seq_lengths,
        'labels': labels
    }