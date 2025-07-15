import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torch.cuda.amp import autocast

class TextClassifier(pl.LightningModule):
    """基于Transformer的文本分类模型"""
    
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", lr: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # 骨干网络
        self.encoder = AutoModel.from_pretrained(model_name)
        # 冻结预训练参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 分类头
        hid_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hid_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask):
        # 自动设备管理
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 混合精度支持
        with autocast(enabled=self.training and torch.cuda.is_available()):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled = outputs.last_hidden_state[:, 0]
            return self.classifier(pooled)
    
    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, mask = batch
        logits = self(x, mask)
        return torch.softmax(logits, dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=2,
            mode='min'
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

#import torch
#import torch.nn as nn
#import pytorch_lightning as pl
#from transformers import AutoModel
#
#class TextClassifier(pl.LightningModule):
#    """
#    A simple text classification model based on a HuggingFace Transformer + MLP head.
#    """
#
#    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", lr: float = 1e-5):
#        super().__init__()
#        self.save_hyperparameters()
#        # Backbone transformer
#        self.encoder = AutoModel.from_pretrained(model_name)
#        # Classification head
#        hid_size = self.encoder.config.hidden_size
#        self.classifier = nn.Sequential(
#            nn.Linear(hid_size, 256),
#            nn.ReLU(),
#            nn.Dropout(0.2),
#            nn.Linear(256, num_classes),
#        )
#        self.loss_fn = nn.CrossEntropyLoss()
#
#        # 半精度加速（RTX 3060）
#        if torch.cuda.is_available():
#            self.encoder = self.encoder.to(torch.float16)
#
#    def forward(self, input_ids, attention_mask):
#        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#        # get [CLS] token 向量
#        pooled = outputs.last_hidden_state[:, 0]
#        return self.classifier(pooled)
#
#    def training_step(self, batch, batch_idx):
#        x, y = batch["input_ids"], batch["labels"]
#        mask = batch["attention_mask"]
#        logits = self(x, mask)
#        loss = self.loss_fn(logits, y)
#        self.log("train_loss", loss, prog_bar=True)
#        return loss
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
#        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
#        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
