import torch
from typing import Dict
from transformers import AutoTokenizer
from .task_model import TextClassifier

class ModelInterface:
    """
    Wraps the Lightning text classifier into a simple .predict() API.
    """

    LABEL_MAP = {
        0: "数学",
        1: "物理",
        2: "化学",
        3: "生物",
        4: "其他",
    }

    def __init__(self, checkpoint_path: str, num_classes: int):
        # Load the trained LightningModule
        self.model = TextClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.hparams.model_name)

    def _preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        }

    def predict(self, text: str) -> Dict:
        """
        Returns: {
          "label": str,
          "confidence": float,
          "is_relevant": bool
        }
        """
        inputs = self._preprocess(text)
        with torch.no_grad():
            logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            max_prob, pred_idx = torch.max(probs, dim=-1)
            label = self.LABEL_MAP[pred_idx.item()]

        return {
            "label": label,
            "confidence": max_prob.item(),
            "is_relevant": max_prob.item() > 0.7
        }

    def is_off_topic(self, text: str) -> bool:
        """
        判断该问题是否超出模型擅长的知识范围
        """
        return not self.predict(text)["is_relevant"]
