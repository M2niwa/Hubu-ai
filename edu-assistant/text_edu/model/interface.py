import torch
from typing import Dict
from transformers import AutoTokenizer
from .task_model import TextClassifier

class ModelInterface:
    LABEL_MAP = {
        0: "数学",
        1: "物理",
        2: "化学",
        3: "生物",
        4: "其他",
    }
    
    def __init__(self, checkpoint_path: str, num_classes: int):
        self.model = TextClassifier.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes
        )
        self.model.eval()
        self.model.freeze()
        
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.hparams.model_name
        )
    
    def _preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return enc
    
    def predict(self, text: str) -> Dict:
        """文本分类预测"""
        inputs = self._preprocess(text)

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=-1)
            max_prob, pred_idx = torch.max(probs, dim=1)
        
        return {
            "label": self.LABEL_MAP[pred_idx.item()],
            "confidence": max_prob.item(),
            "is_relevant": max_prob.item() > 0.7
        }
    
    def is_off_topic(self, text: str) -> bool:
        """判断是否超出知识范围"""
        return not self.predict(text)["is_relevant"]
