import cv2
import os

# 변경할 것
mask_dir = 'DATA/MSD/train/mask'
output_dir = 'DATA/MSD/train/edge'

# 결과 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 파일 목록 가져오기
mask_files = os.listdir(mask_dir)

for mask_file in mask_files:
    if mask_file.endswith('.png') or mask_file.endswith('.jpg'):  # 이미지 파일만 처리
        # 마스크 이미지 읽기
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Canny edge 검출
        edges = cv2.Canny(mask, 100, 200)  # 임계값은 조정 가능

        # 결과 저장
        output_path = os.path.join(output_dir, f'{mask_file}')
        cv2.imwrite(output_path, edges)

print("Edge 추출이 완료되었습니다.")
