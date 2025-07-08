import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import json
from sklearn.model_selection import train_test_split

# 거울 탐지 데이터셋 정보
MIRROR_DATASETS = {
    'MSD': {
        'url': 'https://drive.google.com/file/d/1Znw92fO6lCKfXejjSSyMyL1qtFepgjPI/view',
        'images': 4018,
        'train': 3063,
        'test': 955,
        'description': 'Mirror Segmentation Dataset - 가장 많이 사용되는 기본 데이터셋'
    },
    'PMD': {
        'url': 'https://github.com/Mhaiyang/CVPR2020_Progressive-Mirror-Detection',
        'images': 6461,
        'train': 5168,
        'test': 1293,
        'description': 'Progressive Mirror Detection Dataset - 더 다양하고 큰 규모'
    },
    'RGBD-Mirror': {
        'url': 'https://github.com/Catherine-R-He/RGBD-Mirror',
        'images': 3049,
        'description': 'RGB-D 정보 포함, depth 정보 활용 가능'
    }
}


class MirrorDataset(Dataset):
    """거울 탐지 데이터셋 클래스"""

    def __init__(self, root_dir, split='train', transform=None, dataset_type='MSD'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.dataset_type = dataset_type

        # 이미지와 마스크 경로 설정
        self.img_dir = os.path.join(root_dir, split, 'image')
        self.mask_dir = os.path.join(root_dir, split, 'mask')

        # 파일 리스트 생성
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])

        print(f"Loaded {len(self.images)} {split} images from {dataset_type}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 이미지 로드
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 마스크 로드
        mask_name = img_name.replace('.jpg', '.png')  # 대부분 마스크는 PNG
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # 그레이스케일

        # 전처리
        if self.transform:
            # 동일한 random seed로 이미지와 마스크에 같은 변환 적용
            seed = np.random.randint(2147483647)

            torch.manual_seed(seed)
            image = self.transform['image'](image)

            torch.manual_seed(seed)
            mask = self.transform['mask'](mask)

            # 마스크 이진화
            mask = (mask > 0.5).float()

        return image, mask, img_name


# 데이터 증강 및 전처리
def get_transforms(img_size=384, is_train=True):
    """거울 탐지에 최적화된 데이터 증강"""

    # 이미지 전처리
    if is_train:
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 마스크 전처리
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    return {'image': img_transform, 'mask': mask_transform}


# 다중 데이터셋 결합
class CombinedMirrorDataset(Dataset):
    """여러 거울 탐지 데이터셋을 결합"""

    def __init__(self, dataset_configs, split='train', transform=None):
        self.datasets = []
        self.dataset_lengths = []

        for config in dataset_configs:
            dataset = MirrorDataset(
                root_dir=config['root_dir'],
                split=split,
                transform=transform,
                dataset_type=config['type']
            )
            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))

        self.total_length = sum(self.dataset_lengths)
        print(f"Combined dataset: {self.total_length} {split} images")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 어느 데이터셋에서 가져올지 결정
        dataset_idx = 0
        sample_idx = idx

        for i, length in enumerate(self.dataset_lengths):
            if sample_idx < length:
                dataset_idx = i
                break
            sample_idx -= length

        return self.datasets[dataset_idx][sample_idx]


# 데이터셋 다운로드 및 준비 스크립트
def prepare_datasets(base_dir='../DATA'):
    """데이터셋 다운로드 및 구조 확인"""
    os.makedirs(base_dir, exist_ok=True)


# 데이터셋 통계 및 시각화
def analyze_dataset(dataset_path):
    """데이터셋 분석 및 통계"""
    stats = {
        'total_images': 0,
        'total_masks': 0,
        'image_sizes': [],
        'mirror_ratios': []
    }

    for split in ['train', 'test']:
        img_dir = os.path.join(dataset_path, split, 'image')
        mask_dir = os.path.join(dataset_path, split, 'mask')

        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
            stats['total_images'] += len(images)

            # 샘플 이미지 분석
            for img_name in images[:100]:  # 처음 100개만 분석
                img = cv2.imread(os.path.join(img_dir, img_name))
                if img is not None:
                    stats['image_sizes'].append(img.shape[:2])

                # 마스크 분석
                mask_name = img_name.replace('.jpg', '.png')
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, 0)
                    if mask is not None:
                        mirror_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                        stats['mirror_ratios'].append(mirror_ratio)

    # 통계 출력
    print(f"\n데이터셋 분석 결과:")
    print(f"- 총 이미지 수: {stats['total_images']}")
    print(f"- 평균 이미지 크기: {np.mean(stats['image_sizes'], axis=0) if stats['image_sizes'] else 'N/A'}")
    print(f"- 평균 거울 영역 비율: {np.mean(stats['mirror_ratios']):.2%}" if stats['mirror_ratios'] else 'N/A')

    return stats

'''
# 학습/검증 분할 (PMD 데이터셋용)
def split_dataset(image_dir, mask_dir, output_dir, val_ratio=0.2):
    """데이터셋을 학습/검증으로 분할"""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    # 분할
    train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)

    # 디렉토리 생성
    splits = {'train': train_imgs, 'val': val_imgs}

    for split, img_list in splits.items():
        # 이미지 디렉토리
        img_out_dir = os.path.join(output_dir, split, 'image')
        mask_out_dir = os.path.join(output_dir, split, 'mask')
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        # 파일 복사
        for img_name in img_list:
            # 이미지 복사
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(img_out_dir, img_name)
            os.symlink(src_img, dst_img)  # 심볼릭 링크로 공간 절약

            # 마스크 복사
            mask_name = img_name.replace('.jpg', '.png')
            src_mask = os.path.join(mask_dir, mask_name)
            dst_mask = os.path.join(mask_out_dir, mask_name)
            if os.path.exists(src_mask):
                os.symlink(src_mask, dst_mask)

    print(f"데이터셋 분할 완료: Train {len(train_imgs)}, Val {len(val_imgs)}")
'''

# 사용 예시
if __name__ == "__main__":
    # 1. 데이터셋 준비
    prepare_datasets('../DATA')

    # 2. 데이터셋 분석
    # analyze_dataset('../data/MSD')

    # 3. 데이터로더 생성
    transforms_train = get_transforms(img_size=384, is_train=True)
    transforms_val = get_transforms(img_size=384, is_train=False)

    # 단일 데이터셋 사용
    train_dataset = MirrorDataset(
        root_dir='../DATA/MSD',
        split='train',
        transform=transforms_train,
        dataset_type='MSD'
    )

    # 또는 다중 데이터셋 결합
    # dataset_configs = [
    #     {'root_dir': './data/MSD', 'type': 'MSD'},
    #     {'root_dir': './data/PMD', 'type': 'PMD'}
    # ]
    # train_dataset = CombinedMirrorDataset(
    #     dataset_configs=dataset_configs,
    #     split='train',
    #     transform=transforms_train
    # )

    # 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 샘플 확인
    for i, (images, masks, names) in enumerate(train_loader):
        print(f"Batch {i}: Images {images.shape}, Masks {masks.shape}")
        if i >= 2:
            break