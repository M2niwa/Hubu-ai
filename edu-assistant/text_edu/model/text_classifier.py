import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torch.cuda.amp import autocast

class TextClassifier(pl.LightningModule):
    """基于预训练Transformer的文本分类模型"""
    
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", lr: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # 骨干网络 - 预训练模型
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 冻结编码器参数 (小样本场景建议)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 分类头
        hid_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hid_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask):
        # 确保输入数据在正确设备上
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 自动混合精度处理
        with autocast(enabled=self.training and torch.cuda.is_available()):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 使用[CLS]标记进行分类
            pooled = outputs.last_hidden_state[:, 0]
            return self.classifier(pooled)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=self.hparams.lr
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        return probs
