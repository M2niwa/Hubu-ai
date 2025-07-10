import torch
import os
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    BertTokenizer
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from config import config
from data.dataset import TextDataset, load_data
from models.bert_model import BertClassifier
from utils.logger import init_logger
from utils.metrics import calculate_metrics

def main():
    logger = init_logger()
    logger.info("***** 训练开始 *****")

    device = torch.device(config.device)
    amp_enabled = device.type == 'cuda'
    logger.info(f"使用设备: {device} | 混合精度启用: {amp_enabled}")

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    model = BertClassifier.from_pretrained(
        config.pretrained_model,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True  # 允许尺寸不匹配
    )
    model.to(device)
    logger.info(f"成功加载预训练模型: {config.pretrained_model}")

    train_texts, val_texts, train_labels, val_labels = load_data(config.data_path)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, config.max_seq_len)
    train_loader = DataLoader(train_dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             pin_memory=True)

    val_loader = None
    if len(val_texts) > 0:
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_seq_len)
        val_loader = DataLoader(val_dataset,
                               batch_size=config.batch_size,
                               shuffle=False,
                               pin_memory=True)
        logger.info(f"验证集加载成功，样本数: {len(val_texts)}")
    else:
        logger.warning("未找到验证集，将跳过验证步骤")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):  # 更新autocast用法
                outputs = model(**inputs, labels=labels)
                loss = outputs['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_train_loss / (step + 1)
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch + 1} | Step {step}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

        if val_loader:
            model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(device)

                    with autocast(enabled=amp_enabled):
                        outputs = model(**inputs, labels=labels)

                    total_val_loss += outputs['loss'].item()
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().numpy())
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
                save_path = os.path.join(config.save_dir, f"best_model_epoch{epoch + 1}.bin")
                torch.save(model.state_dict(), save_path)
                logger.info(f"保存最佳模型到: {save_path}")
        else:
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} 训练损失: {avg_train_loss:.4f}")

if __name__ == "__main__":
    main()