import os
import pandas as pd
from typing import List, Tuple
from config import constants

class DataLoader:
    """统一数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.allowed_ext = constants.ALLOWED_EXTENSIONS
    
    def load_texts(self) -> List[Tuple[str, int]]:
        """加载所有文本数据"""
        texts = []
        labels = []
        
        # 遍历目录结构
        for label_dir in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            
            label_id = self._get_label_id(label_dir)
            for file in os.listdir(label_path):
                if any(file.endswith(ext) for ext in self.allowed_ext):
                    file_path = os.path.join(label_path, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(label_id)
        
        return texts, labels
    
    def _get_label_id(self, label_name: str) -> int:
        """标签名称转ID"""
        label_map = {
            "数学": 0,
            "物理": 1,
            "化学": 2,
            "生物": 3,
            "其他": 4
        }
        return label_map.get(label_name, 4)
