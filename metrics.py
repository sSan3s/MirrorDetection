# training/metrics.py
"""
HetNet 원본과 호환되는 평가 메트릭
misc.py를 기반으로 PyTorch와 통합
"""

import torch
import numpy as np
import os
from PIL import Image
import skimage.io
import skimage.transform


class AvgMeter(object):
    """평균값 추적을 위한 헬퍼 클래스"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_size(eval_segm, gt_segm):
    """두 세그멘테이션 맵의 크기가 같은지 확인"""
    h_e, w_e = eval_segm.shape[:2]
    h_g, w_g = gt_segm.shape[:2]

    if (h_e != h_g) or (w_e != w_g):
        raise ValueError(f"Different dimensions: pred {h_e}x{w_e} vs gt {h_g}x{w_g}")


def accuracy_mirror(predict_mask, gt_mask):
    """
    거울 영역에 대한 정확도 (TP/N_p)
    HetNet 원본 구현
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)  # 실제 positive 픽셀 수
    if N_p == 0:
        return 0.0

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    accuracy_ = TP / N_p

    return accuracy_


def accuracy_image(predict_mask, gt_mask):
    """
    전체 이미지에 대한 픽셀 정확도 (TP+TN)/(N_p+N_n)
    HetNet 원본 구현
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = (TP + TN) / (N_p + N_n)

    return accuracy_


def compute_iou(predict_mask, gt_mask):
    """
    Intersection over Union 계산
    HetNet 원본 구현
    """
    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        return 0.0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))  # intersection
    t_i = np.sum(gt_mask)  # ground truth 영역
    n_ij = np.sum(predict_mask)  # prediction 영역

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def cal_precision_recall(predict_mask, gt_mask):
    """
    255개의 threshold에서 precision과 recall 계산
    HetNet 원본 구현

    Args:
        predict_mask: 0-255 범위의 uint8 numpy array
        gt_mask: 0-255 범위의 uint8 numpy array
    """
    assert predict_mask.dtype == np.uint8
    assert gt_mask.dtype == np.uint8
    assert predict_mask.shape == gt_mask.shape

    eps = 1e-4

    prediction = predict_mask / 255.
    gt = gt_mask / 255.

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt).astype(np.float32)

    precision, recall = [], []

    # 255개의 다른 이진화 threshold에서 precision과 recall 계산
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt).astype(np.float32)
        p = np.sum(hard_prediction).astype(np.float32)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall


def cal_fmeasure(precision, recall):
    """
    최대 F-measure 계산
    HetNet 원본 구현
    """
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def compute_mae(predict_mask, gt_mask):
    """
    Mean Absolute Error 계산
    HetNet 원본 구현
    """
    check_size(predict_mask, gt_mask)
    mae_ = np.mean(abs(predict_mask - gt_mask)).item()
    return mae_


def compute_ber(predict_mask, gt_mask):
    """
    Balance Error Rate 계산
    HetNet 원본 구현
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    if N_p == 0 or N_n == 0:
        return 0.0

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    return ber_


# PyTorch 텐서와의 통합을 위한 래퍼 함수들
def torch_to_numpy_uint8(tensor, sigmoid=False):
    """
    PyTorch 텐서를 0-255 범위의 numpy uint8로 변환

    Args:
        tensor: PyTorch tensor (B, 1, H, W) or (1, H, W) or (H, W)
        sigmoid: sigmoid 적용 여부
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    if sigmoid:
        tensor = torch.sigmoid(tensor)

    # 0-255 범위로 변환
    numpy_array = (tensor * 255).cpu().numpy().astype(np.uint8)
    return numpy_array


def torch_to_numpy_binary(tensor, threshold=0.5, sigmoid=False):
    """
    PyTorch 텐서를 이진 numpy array로 변환

    Args:
        tensor: PyTorch tensor
        threshold: 이진화 threshold
        sigmoid: sigmoid 적용 여부
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    if sigmoid:
        tensor = torch.sigmoid(tensor)

    binary_array = (tensor > threshold).cpu().numpy().astype(np.float32)
    return binary_array


class MirrorMetrics:
    """
    HetNet 스타일의 거울 탐지 메트릭 계산 클래스
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """메트릭 초기화"""
        self.accuracy_mirror_meter = AvgMeter()
        self.accuracy_image_meter = AvgMeter()
        self.iou_meter = AvgMeter()
        self.mae_meter = AvgMeter()
        self.ber_meter = AvgMeter()
        self.fmeasure_meter = AvgMeter()

    def update(self, predict, target, sigmoid_applied=False):
        """
        배치에 대한 메트릭 업데이트

        Args:
            predict: 모델 출력 (B, 1, H, W) PyTorch tensor
            target: Ground truth (B, 1, H, W) PyTorch tensor (0 or 1)
            sigmoid_applied: predict에 sigmoid가 이미 적용되었는지 여부
        """
        batch_size = predict.size(0)

        for i in range(batch_size):
            # 개별 샘플 추출
            pred_i = predict[i]
            target_i = target[i]

            # 이진 마스크 (IoU, accuracy 등용)
            pred_binary = torch_to_numpy_binary(pred_i, sigmoid=not sigmoid_applied)
            target_binary = torch_to_numpy_binary(target_i, sigmoid=False)

            # 0-255 범위 마스크 (F-measure, MAE 등용)
            pred_uint8 = torch_to_numpy_uint8(pred_i, sigmoid=not sigmoid_applied)
            target_uint8 = torch_to_numpy_uint8(target_i, sigmoid=False)

            # 0-1 범위 마스크 (MAE용)
            pred_01 = pred_uint8 / 255.0
            target_01 = target_uint8 / 255.0

            # 각 메트릭 계산
            self.accuracy_mirror_meter.update(accuracy_mirror(pred_binary, target_binary))
            self.accuracy_image_meter.update(accuracy_image(pred_binary, target_binary))
            self.iou_meter.update(compute_iou(pred_binary, target_binary))
            self.mae_meter.update(compute_mae(pred_01, target_01))
            self.ber_meter.update(compute_ber(pred_binary, target_binary))

            # F-measure 계산
            precision, recall = cal_precision_recall(pred_uint8, target_uint8)
            fmeasure = cal_fmeasure(precision, recall)
            self.fmeasure_meter.update(fmeasure)

    def get_results(self):
        """평균 메트릭 반환"""
        return {
            'accuracy_mirror': self.accuracy_mirror_meter.avg,
            'accuracy_image': self.accuracy_image_meter.avg,
            'iou': self.iou_meter.avg,
            'mae': self.mae_meter.avg,
            'ber': self.ber_meter.avg,
            'f_measure': self.fmeasure_meter.avg,
            'dice': 2 * self.iou_meter.avg / (1 + self.iou_meter.avg)  # IoU에서 Dice 계산
        }

    def print_results(self):
        """결과 출력 (HetNet 스타일)"""
        results = self.get_results()
        print("\n" + "="*50)
        print("Mirror Detection Metrics (HetNet Style)")
        print("="*50)
        print(f"Accuracy (Mirror): {results['accuracy_mirror']:.4f}")
        print(f"Accuracy (Image):  {results['accuracy_image']:.4f}")
        print(f"IoU:               {results['iou']:.4f}")
        print(f"Dice:              {results['dice']:.4f}")
        print(f"F-measure:         {results['f_measure']:.4f}")
        print(f"MAE:               {results['mae']:.4f}")
        print(f"BER:               {results['ber']:.4f}")
        print("="*50 + "\n")


# 단일 이미지 평가를 위한 헬퍼 함수
def evaluate_single_image(pred_path, gt_path):
    """
    단일 이미지 쌍에 대한 평가

    Args:
        pred_path: 예측 마스크 경로
        gt_path: Ground truth 마스크 경로
    """
    # 이미지 로드
    pred_mask = skimage.io.imread(pred_path)
    gt_mask = skimage.io.imread(gt_path)

    # 0-255 범위로 정규화
    if pred_mask.max() <= 1:
        pred_mask = (pred_mask * 255).astype(np.uint8)
    if gt_mask.max() <= 1:
        gt_mask = (gt_mask * 255).astype(np.uint8)

    # 이진 마스크
    pred_binary = (pred_mask > 127).astype(np.float32)
    gt_binary = (gt_mask > 127).astype(np.float32)

    # 0-1 범위
    pred_01 = pred_mask / 255.0
    gt_01 = gt_mask / 255.0

    # 메트릭 계산
    results = {
        'accuracy_mirror': accuracy_mirror(pred_binary, gt_binary),
        'accuracy_image': accuracy_image(pred_binary, gt_binary),
        'iou': compute_iou(pred_binary, gt_binary),
        'mae': compute_mae(pred_01, gt_01),
        'ber': compute_ber(pred_binary, gt_binary),
    }

    # F-measure
    precision, recall = cal_precision_recall(pred_mask, gt_mask)
    results['f_measure'] = cal_fmeasure(precision, recall)
    results['dice'] = 2 * results['iou'] / (1 + results['iou'])

    return results


# 사용 예시
if __name__ == "__main__":
    # PyTorch 텐서로 테스트
    pred = torch.rand(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()

    # 메트릭 계산
    metrics = MirrorMetrics()
    metrics.update(pred, target, sigmoid_applied=False)
    metrics.print_results()

    print("\n개별 메트릭 테스트:")
    # 개별 numpy array로 테스트
    pred_np = torch.sigmoid(pred[0, 0]).numpy()
    target_np = target[0, 0].numpy()

    print(f"IoU: {compute_iou(pred_np > 0.5, target_np):.4f}")
    print(f"MAE: {compute_mae(pred_np, target_np):.4f}")