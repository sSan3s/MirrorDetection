#!/usr/bin/env python3
"""
MViTv2 기반 거울 탐지 모델 학습 파이프라인 (wandb 선택적 사용)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import json
import logging


# 설정
class TrainingConfig:
    # 모델 설정
    model_variant = 'base'  # tiny, small, base, large
    pretrained = False
    pretrained_21k = False  # ImageNet-21K 사용 여부

    # 학습 설정
    batch_size = 16
    accumulation_steps = 1  # Gradient accumulation
    num_epochs = 150
    warmup_epochs = 1

    # 옵티마이저
    optimizer = 'adamw'
    base_lr = 3e-4  # MViTv2는 CNN보다 낮은 lr 필요
    weight_decay = 0.01

    # 스케줄러
    scheduler = 'cosine'
    min_lr = 1e-6

    # 데이터
    img_size = 384
    num_workers = 4

    # 기타
    amp = True  # Automatic Mixed Precision
    gradient_clip = 1.0
    save_freq = 5
    eval_freq = 1

    # 경로
    data_root = '../DATA/MSD'
    checkpoint_dir = './checkpoints/mvitv2_mirror'
    log_dir = './logs'


# 로깅 설정
def setup_logger(name, log_file, level=logging.INFO):
    """로거 설정"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # 콘솔 출력도 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class MirrorDetectionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 디렉토리 생성
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # 로깅 설정
        self.setup_logging()

        # 모델, 옵티마이저, 스케줄러
        self.model = build_model(config).to(self.device)
        self.optimizer = get_optimizer(self.model, config)

        # 데이터로더
        from mirror_dataset_prep import MirrorDataset, get_transforms

        transforms_train = get_transforms(config.img_size, is_train=True)
        transforms_val = get_transforms(config.img_size, is_train=False)

        train_dataset = MirrorDataset(config.data_root, 'train', transforms_train)
        val_dataset = MirrorDataset(config.data_root, 'test', transforms_val)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        steps_per_epoch = len(self.train_loader) // config.accumulation_steps
        self.scheduler = get_scheduler(self.optimizer, config, steps_per_epoch)

        # 손실 함수
        self.criterion = self.get_loss_function()

        # Mixed Precision
        self.scaler = GradScaler() if config.amp else None

        self.start_epoch = 0
        self.best_iou = 0

    def setup_logging(self):
        """로깅 설정 (wandb 선택적 사용)"""
        log_name = f"mvitv2_{self.config.model_variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 로거 설정
        log_file = os.path.join(self.config.log_dir, f'{log_name}.log')
        self.logger = setup_logger('MirrorDetection', log_file)
        self.logger.info(f"Training configuration: {self.config.__dict__}")

        # TensorBoard 설정
        self.tb_writer = SummaryWriter(os.path.join(self.config.log_dir, 'tensorboard', log_name))

        # 로컬 JSON 로그
        self.json_log_file = os.path.join(self.config.log_dir, f'{log_name}_history.json')
        self.training_history = []


    def log_metrics(self, metrics, epoch):
        """메트릭 로깅 (멀티 백엔드)"""
        # TensorBoard
        for key, value in metrics.items():
            if key != 'epoch':
                self.tb_writer.add_scalar(key, value, epoch)

        # Local JSON
        self.training_history.append(metrics)
        with open(self.json_log_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Logger
        self.logger.info(f"Epoch {epoch}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    def get_loss_function(self):
        """복합 손실 함수"""
        from losses import MirrorLoss
        return MirrorLoss()

    def train_epoch(self, epoch):
        """한 에폭 학습"""
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs}')
        total_loss = 0
        loss_components = {'bce': 0, 'dice': 0, 'iou': 0, 'ssim': 0}

        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Mixed Precision Training
            with autocast(enabled=self.config.amp):
                outputs = self.model(images)
                loss, components = self.criterion(outputs, masks)
                loss = loss / self.config.accumulation_steps

            # Backward
            if self.config.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                if self.config.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

            # 로깅
            total_loss += loss.item() * self.config.accumulation_steps
            for k, v in components.items():
                if k in loss_components:
                    loss_components[k] += v

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        # 에폭 평균
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_components:
            loss_components[k] /= len(self.train_loader)

        return avg_loss, loss_components

    def validate(self):
        """검증"""
        self.model.eval()

        from metrics import MirrorMetrics
        metrics_calculator = MirrorMetrics()

        with torch.no_grad():
            for images, masks, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = torch.sigmoid(self.model(images))
                metrics_calculator.update(outputs, masks)

        return metrics_calculator.get_results()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'best_iou': self.best_iou
        }

        filename = 'best.pth' if is_best else f'epoch_{epoch + 1}.pth'
        save_path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)

        self.logger.info(f"Saved checkpoint: {save_path}")

    def train(self):
        """전체 학습 루프"""
        self.logger.info("Starting training...")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            # 학습
            train_loss, loss_components = self.train_epoch(epoch)

            # 검증
            if (epoch + 1) % self.config.eval_freq == 0:
                val_metrics = self.validate()

                # 메트릭 정리
                metrics = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/bce': loss_components.get('bce', 0),
                    'train/dice': loss_components.get('dice', 0),
                    'train/iou': loss_components.get('iou', 0),
                    'train/ssim': loss_components.get('ssim', 0),
                    'val/iou': val_metrics['iou'],
                    'val/dice': val_metrics['dice'],
                    'val/mae': val_metrics['mae'],
                    'val/f_measure': val_metrics.get('f_measure', 0),
                    'lr': self.scheduler.get_last_lr()[0]
                }

                # 로깅
                self.log_metrics(metrics, epoch + 1)

                # 모델 저장
                if val_metrics['iou'] > self.best_iou:
                    self.best_iou = val_metrics['iou']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)

            # 정기 저장
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch, None, is_best=False)

        # 학습 완료
        self.tb_writer.close()
        self.logger.info("Training completed!")

        # 최종 결과 출력
        self.logger.info(f"Best IoU: {self.best_iou:.4f}")


# 나머지 함수들 (build_model, get_optimizer, get_scheduler)은 이전과 동일
def build_model(config):
    """MViTv2 기반 HetNet 모델 생성"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from hetnet_mvitv2 import HetNetMViTv2

    # 모델 생성
    model = HetNetMViTv2(num_classes=1)


    return model


def get_optimizer(model, config):
    """옵티마이저 생성"""
    # ... (이전 코드와 동일)
    param_groups = []
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': config.base_lr * 0.1},
        {'params': head_params, 'lr': config.base_lr}
    ]

    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

    return optimizer


def get_scheduler(optimizer, config, steps_per_epoch):
    """학습률 스케줄러 생성"""
    # ... (이전 코드와 동일)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps

        if config.scheduler == 'cosine':
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        else:
            return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# 메인 실행
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='./data/MSD')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # 설정 업데이트
    config = TrainingConfig()
    config.model_variant = args.variant
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.base_lr = args.lr
    config.data_root = args.data_root


    # 학습 시작
    trainer = MirrorDetectionTrainer(config)

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch']
        trainer.best_iou = checkpoint.get('best_iou', 0)
        print(f"Resumed from epoch {trainer.start_epoch}")

    trainer.train()


if __name__ == "__main__":
    main()