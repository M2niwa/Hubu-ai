import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from config import config
from data.lstm_dataset import TextDataset, load_data, collate_fn
from models.lstm_model import BiLSTMModel
from utils.logger import init_logger
from utils.metrics import calculate_metrics

def main():
    logger = init_logger()
    logger.info("***** LSTM 训练开始 *****")

    device = torch.device(config.device)
    logger.info(f"使用设备: {device}")

    # 加载数据并构建词汇表
    train_texts, val_texts, train_labels, val_labels, word_to_idx = load_data(
        data_path=config.data_path,
        max_vocab_size=config.max_vocab_size,
        max_seq_len=config.max_seq_len
    )
    logger.info(f"词汇表大小: {len(word_to_idx)}")

    # 创建数据集和DataLoader
    train_dataset = TextDataset(train_texts, train_labels, word_to_idx, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = None
    if val_texts:
        val_dataset = TextDataset(val_texts, val_labels, word_to_idx, config.max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True
        )
        logger.info(f"验证集样本数: {len(val_texts)}")
    else:
        logger.warning("未找到验证集，将跳过验证步骤")

    # 初始化模型
    model = BiLSTMModel(config)
    model.to(device)
    logger.info("模型初始化完成")

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, seq_lengths)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_train_loss / (step + 1)
                logger.info(
                    f"Epoch {epoch + 1} | Step {step}/{len(train_loader)} | Loss: {avg_loss:.4f}"
                )

        # 验证步骤
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    seq_lengths = batch['seq_lengths'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, seq_lengths)
                    loss = criterion(outputs, labels)

                    total_val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            metrics = calculate_metrics(all_preds, all_labels)

            logger.info(
                f"Epoch {epoch + 1} 结果 || "
                f"训练损失: {avg_train_loss:.4f} | "
                f"验证损失: {avg_val_loss:.4f} | "
                f"准确率: {metrics['accuracy']:.4f} | "
                f"F1分数: {metrics['f1_score']:.4f}"
            )

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                os.makedirs(config.save_dir, exist_ok=True)
                save_path = os.path.join(config.save_dir, f"best_lstm_model_epoch{epoch + 1}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"保存最佳模型到: {save_path}")
        else:
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} 训练损失: {avg_train_loss:.4f}")

if __name__ == "__main__":
    main()