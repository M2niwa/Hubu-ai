import pytorch_lightning as pl
import torch
import torch.nn as nn

class TextClassifier(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=3
        )
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        logits = self.classifier(z[:, 0])  # 取[CLS]向量
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)