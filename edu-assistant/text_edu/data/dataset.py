import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

class TextDataset(Dataset):
    def __init__(self, texts: list, labels: list, max_len: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        texts: list,
        labels: list,
        batch_size: int = 32,
        max_len: int = 512,
        val_split: float = 0.2,
        seed: int = 42
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.max_len = max_len
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage=None):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.texts, self.labels, test_size=self.val_split, random_state=self.seed
        )
        self.train_dataset = TextDataset(train_texts, train_labels, self.max_len)
        self.val_dataset = TextDataset(val_texts, val_labels, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# 假设已有文本和标签列表
#exts = ["This is great!", "Terrible experience.", "Neutral feedback."]
#labels = [1, 0, 2]  # 对应类别标签

# 构建数据模块
#dm = TextDataModule(texts, labels, batch_size=16)
#dm.setup()

# 加载训练数据
#train_loader = dm.train_dataloader()
#for batch in train_loader:
#    print(batch["input_ids"].shape, batch["labels"])
#    break
