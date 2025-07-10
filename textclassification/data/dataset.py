import json
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = re.sub(r'\d+', '[NUM]', text)
    return text

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [clean_text(t) for t in texts]
        self.labels = labels  # 注意：此处labels应已经是数字形式
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """加载TNEWS数据集并转换标签"""
    # 标签映射（根据文档14）
    label_mapping = {
        "100": 0, "101": 1, "102": 2, "103": 3, "104": 4,
        "106": 5, "107": 6, "108": 7, "109": 8, "110": 9,
        "112": 10, "113": 11, "115": 12, "116": 13, "114": 14
    }

    train_texts, train_labels = [], []
    train_path = os.path.join(data_path, 'train.json').replace('\\', '/')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练文件路径错误：{train_path}")

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            train_texts.append(item['sentence'])
            # 转换标签为数字
            train_labels.append(label_mapping[item['label']])

    val_texts, val_labels = [], []
    dev_path = os.path.join(data_path, 'dev.json').replace('\\', '/')
    if os.path.exists(dev_path):
        with open(dev_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                val_texts.append(item['sentence'])
                val_labels.append(label_mapping[item['label']])

    return train_texts, val_texts, train_labels, val_labels