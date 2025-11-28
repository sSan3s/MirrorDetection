# coding=utf-8

import os
import time
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np

# ★★★ matplotlib을 non-GUI 백엔드로 설정 (Qt 오류 해결) ★★★
import matplotlib

matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset as dataset

from Net_Base_HCO_v2 import Net

from misc import (
    AvgMeter,
    compute_iou,
    compute_mae,
    compute_ber,
    accuracy_image,
    accuracy_mirror,
    cal_precision_recall,
    cal_fmeasure
)


def resize_to_match(pred, target_shape):
    if pred.shape == target_shape:
        return pred
    pred_resized = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return pred_resized


def save_intermediate_visualizations(outputs, edge, mask, image_name, save_dir, dataset_path):
    """
    중간 출력들을 시각화하여 저장
    """
    main_out, out1, out2, out3, out4 = outputs

    # 디렉토리 생성
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # ★★★ 원본 RGB 이미지 로드 ★★★
    rgb_path = os.path.join(dataset_path, 'test/image', f'{image_name}.jpg')

    rgb_image = cv2.imread(rgb_path)
    if rgb_image is not None:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # mask와 같은 크기로 리사이즈
        if rgb_image.shape[:2] != mask.shape:
            rgb_image = cv2.resize(rgb_image, (mask.shape[1], mask.shape[0]))
        rgb_image = rgb_image / 255.0  # normalize to [0, 1]
    else:
        rgb_image = np.zeros((*mask.shape, 3))  # fallback

    # ★★★ uint8로 명시적 변환하여 저장 (depth 오류 해결) ★★★
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_rgb.png'),
                (rgb_image[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_main.png'),
                np.round(main_out * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_edge.png'),
                np.round(edge * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_stage1.png'),
                np.round(out1 * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_stage2.png'),
                np.round(out2 * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_stage3.png'),
                np.round(out3 * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(vis_dir, f'{image_name}_stage4.png'),
                np.round(out4 * 255).astype(np.uint8))

    # ★★★ 통합 시각화 저장 (2x4 레이아웃) ★★★
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Intermediate Outputs - {image_name}', fontsize=16)

    # Row 0
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original RGB', fontweight='bold', color='blue')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(main_out, cmap='gray')
    axes[0, 2].set_title('Main Output', fontweight='bold', color='red')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(edge, cmap='gray')
    axes[0, 3].set_title('Edge Output', fontweight='bold')
    axes[0, 3].axis('off')

    # Row 1
    axes[1, 0].imshow(out1, cmap='gray')
    axes[1, 0].set_title('Stage 1 (Fused1)', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(out2, cmap='gray')
    axes[1, 1].set_title('Stage 2 (Fused2)', fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(out3, cmap='gray')
    axes[1, 2].set_title('Stage 3 (Fused3)', fontweight='bold')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(out4, cmap='gray')
    axes[1, 3].set_title('Stage 4 (Fused4)', fontweight='bold')
    axes[1, 3].axis('off')


    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{image_name}_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()  # 메모리 누수 방지


def print_metrics_table(metrics_dict):
    try:
        from tabulate import tabulate
        headers = ["Metric", "Value"]
        data = []
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                data.append([key, f"{value:.4f}"])
            else:
                data.append([key, str(value)])

        print("\n" + "=" * 50)
        print("TEST PERFORMANCE METRICS")
        print("=" * 50)
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print("=" * 50)
    except ImportError:
        print("\n" + "=" * 50)
        print("TEST PERFORMANCE METRICS")
        print("=" * 50)
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                print(f"{key:<20}: {value:.4f}")
            else:
                print(f"{key:<20}: {value}")
        print("=" * 50)


dataset_name = 'MSD'
exp_name = 'check-msd'
VISUALIZE_INTERMEDIATES = True  # 중간 시각화 활성화 플래그


class Test(object):
    def __init__(self, Dataset, Network, path):
        snapshot_path = f'./{exp_name}/model-best'

        self.cfg = Dataset.Config(dataset=dataset_name, datapath=path,
                                  snapshot=snapshot_path, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)

        self.net = Network(self.cfg)

        try:
            print(f"Loading model from: {snapshot_path}")
            state_dict = torch.load(snapshot_path)
            self.net.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized weights.")

        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            cost_time = list()
            mae_meter = AvgMeter()
            iou_meter = AvgMeter()
            ber_meter = AvgMeter()
            acc_meter = AvgMeter()
            mirror_acc_meter = AvgMeter()

            precision_record = [AvgMeter() for _ in range(256)]
            recall_record = [AvgMeter() for _ in range(256)]

            print(f"Starting evaluation on {len(self.loader.dataset)} images...")
            print(f"Intermediate visualization: {'ON' if VISUALIZE_INTERMEDIATES else 'OFF'}")

            for idx, (image, mask, shape, name) in enumerate(self.loader):
                image = image.cuda().float()

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                # ★★★ 핵심 변경: mode를 임시로 'train'으로 설정하여 중간 출력 받기 ★★★
                if VISUALIZE_INTERMEDIATES:
                    original_mode = self.cfg.mode
                    self.cfg.mode = 'train'  # 임시로 train 모드로 변경
                    main_out, edge_out, out1, out2, out3, out4 = self.net(image, shape)
                    self.cfg.mode = original_mode  # 원래 모드로 복구
                else:
                    main_out, edge_out = self.net(image, shape)

                torch.cuda.synchronize()
                cost_time.append(time.perf_counter() - start_time)

                # 예측 결과 변환
                pred = torch.sigmoid(main_out[0, 0]).cpu().numpy()
                mask_np = mask[0].numpy()
                edge = torch.sigmoid(edge_out[0, 0]).cpu().numpy()

                # 크기 불일치 해결
                if pred.shape != mask_np.shape:
                    pred = resize_to_match(pred, mask_np.shape)
                    edge = resize_to_match(edge, mask_np.shape)

                # 기본 결과 저장 (uint8로 명시적 변환)
                save_path = f'./map-{dataset_name}/{dataset_name}/' + self.cfg.datapath.split('/')[-1]
                save_edge = f'./map-{dataset_name}/Edge/' + self.cfg.datapath.split('/')[-1]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(save_edge):
                    os.makedirs(save_edge)

                cv2.imwrite(save_path + '/' + name[0] + '.png',
                            np.round(pred * 255).astype(np.uint8))
                cv2.imwrite(save_edge + '/' + name[0] + '_edge.png',
                            np.round(edge * 255).astype(np.uint8))

                # 중간 출력 시각화 (선택된 이미지들만)
                if VISUALIZE_INTERMEDIATES and (idx < 10 or idx % 50 == 0):
                    # 중간 출력들 변환
                    intermediate_outputs = []
                    for intermediate_out in [out1, out2, out3, out4]:
                        inter_pred = torch.sigmoid(intermediate_out[0, 0]).cpu().numpy()
                        if inter_pred.shape != mask_np.shape:
                            inter_pred = resize_to_match(inter_pred, mask_np.shape)
                        intermediate_outputs.append(inter_pred)

                    # ★★★ dataset_path 전달하여 RGB 이미지 로드 ★★★
                    save_intermediate_visualizations(
                        (pred, *intermediate_outputs),
                        edge,
                        mask_np,
                        name[0],
                        f'./map-{dataset_name}/',
                        self.cfg.datapath  # RGB 이미지 경로
                    )

                # 평가 메트릭 계산
                mae = compute_mae(pred, mask_np)
                mae_meter.update(mae)

                binary_pred = (pred > 0.5).astype(np.float32)
                iou = compute_iou(binary_pred, mask_np)
                iou_meter.update(iou)

                ber = compute_ber(binary_pred, mask_np)
                ber_meter.update(ber)

                acc = accuracy_image(binary_pred, mask_np)
                acc_meter.update(acc)

                mirror_acc = accuracy_mirror(binary_pred, mask_np)
                mirror_acc_meter.update(mirror_acc)

                pred_255 = (pred * 255).astype(np.uint8)
                mask_255 = (mask_np * 255).astype(np.uint8)

                p, r = cal_precision_recall(pred_255, mask_255)

                for idx_pr, (p_val, r_val) in enumerate(zip(p, r)):
                    precision_record[idx_pr].update(p_val)
                    recall_record[idx_pr].update(r_val)

                # 진행률 출력
                if (idx + 1) % 50 == 0 or (idx + 1) == len(self.loader.dataset):
                    print(f"Processed {idx + 1}/{len(self.loader.dataset)} images...")

            # 첫 번째 이미지 실행 시간 제외
            if len(cost_time) > 0:
                cost_time.pop(0)

            # F-measure 계산
            f_measure = cal_fmeasure([p_record.avg for p_record in precision_record],
                                     [r_record.avg for r_record in recall_record])

            # 모델 복잡도 계산
            params_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            params_m = params_count / 1e6

            # 시간 관련 메트릭
            avg_time = np.mean(cost_time) if len(cost_time) > 0 else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0

            # 메트릭 저장
            metrics = {
                "MAE ↓": mae_meter.avg,
                "IoU ↑": iou_meter.avg,
                "BER ↓": ber_meter.avg,
                "Accuracy ↑": acc_meter.avg,
                "Mirror Acc ↑": mirror_acc_meter.avg,
                "F-measure ↑": f_measure,
                "Parameters (M)": params_m,
                "Avg Time (s)": avg_time,
                "FPS": fps,
                "Total Images": len(self.loader.dataset)
            }

            # 결과 출력
            print_metrics_table(metrics)

            # 결과 파일 저장
            result_dir = f'./map-{dataset_name}'
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, 'performance_metrics.txt')

            with open(result_path, 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Model Snapshot: {self.cfg.snapshot}\n")
                f.write(f"Intermediate Visualization: {'ON' if VISUALIZE_INTERMEDIATES else 'OFF'}\n")
                f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "=" * 50 + "\n")
                f.write("PERFORMANCE METRICS\n")
                f.write("=" * 50 + "\n")

                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key:<20}: {value:.4f}\n")
                    else:
                        f.write(f"{key:<20}: {value}\n")

            print(f"\nDetailed results saved to: {result_path}")

            if VISUALIZE_INTERMEDIATES:
                print(f"Intermediate visualizations saved to: {result_dir}/visualizations/")

            return metrics


if __name__ == '__main__':
    for path in [f'../DATA/{dataset_name}/']:
        print(f"Testing on {path}...")
        test = Test(dataset, Net, path)
        results = test.save()
        print(f"\nTesting completed. Results saved in ./map-{dataset_name}/")

        print(f"\n>>> QUICK SUMMARY:")
        print(f"   MAE: {results['MAE ↓']:.4f}")
        print(f"   IoU: {results['IoU ↑']:.4f}")
        print(f"   F-measure: {results['F-measure ↑']:.4f}")
        print(f"   FPS: {results['FPS']:.2f}")