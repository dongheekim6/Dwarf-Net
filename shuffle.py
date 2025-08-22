import os
import shutil
import random
from glob import glob
import numpy as np

# 폴더 경로 설정
images_folder = '/userHome/userhome2/donghee/modelcombination/sessile-main-Kvasir-SEG/images'
masks_folder = '/userHome/userhome2/donghee/modelcombination/sessile-main-Kvasir-SEG/masks'
combined_folder = 'combined_kvasir'

# 1. combined 폴더 생성 및 기존 폴더에서 파일 복사 및 이름 변경
os.makedirs(combined_folder, exist_ok=True)
image_mask_pairs = []

# 이미지와 마스크 파일을 쌍으로 수집
for img_file in sorted(glob(os.path.join(images_folder, '*.jpg'))):  # .jpg 확장자
    img_name = os.path.basename(img_file)
    img_number = os.path.splitext(img_name)[0]
    mask_file = os.path.join(masks_folder, f"{img_number}.jpg")  # 마스크도 .jpg

    if os.path.exists(mask_file):
        img_dest = os.path.join(combined_folder, f"{img_number}.jpg")
        mask_dest = os.path.join(combined_folder, f"{img_number}_mask.jpg")

        shutil.copy(img_file, img_dest)
        shutil.copy(mask_file, mask_dest)

        image_mask_pairs.append((img_dest, mask_dest))

# 총 쌍 수 확인
print(f"\n총 이미지-마스크 쌍 수집: {len(image_mask_pairs)}개")

# 전체 데이터셋 인덱스화
total_samples = len(image_mask_pairs)
all_indices = list(range(total_samples))

# 10개의 서로 다른 split 생성
splits_count = 10
all_split_compositions = []

for split_num in range(splits_count):
    split_folder = f'splits{split_num + 1}'
    os.makedirs(split_folder, exist_ok=True)

    # 시드 고정
    random_seed = np.random.randint(0, 10000)
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(all_indices)

    # 8:1:1 비율
    n_train = round(total_samples * 0.8)
    n_valid = round(total_samples * 0.1)
    n_test = total_samples - n_train - n_valid

    train_indices = set(shuffled_indices[:n_train])
    valid_indices = set(shuffled_indices[n_train:n_train + n_valid])
    test_indices = set(shuffled_indices[n_train + n_valid:])

    current_composition = (train_indices, valid_indices, test_indices)

    # 중복률 계산
    if all_split_compositions:
        overlap_scores = []
        for prev_comp in all_split_compositions:
            train_overlap = len(train_indices.intersection(prev_comp[0])) / n_train
            valid_overlap = len(valid_indices.intersection(prev_comp[1])) / len(valid_indices)
            test_overlap = len(test_indices.intersection(prev_comp[2])) / len(test_indices)
            avg_overlap = (train_overlap + valid_overlap + test_overlap) / 3
            overlap_scores.append(avg_overlap)

        print(f"\nSplit {split_num + 1} 평균 중복률: {np.mean(overlap_scores):.2%}")

    all_split_compositions.append(current_composition)

    # 파일 복사
    for category, indices in zip(['training', 'validation', 'test'],
                                  [train_indices, valid_indices, test_indices]):
        category_folder = os.path.join(split_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        sorted_indices = sorted(indices)

        for idx in sorted_indices:
            img_file, mask_file = image_mask_pairs[idx]

            img_number = os.path.splitext(os.path.basename(img_file))[0]

            new_img_path = os.path.join(category_folder, f"{img_number}.png")
            new_mask_path = os.path.join(category_folder, f"{img_number}_mask.png")

            shutil.copy(img_file, new_img_path)
            shutil.copy(mask_file, new_mask_path)

# split별로 이미지 수 확인
for split_num in range(splits_count):
    split_folder = f'splits{split_num + 1}'
    print(f"\nSplit {split_num + 1} 데이터 수:")
    for category in ['training', 'validation', 'test']:
        category_folder = os.path.join(split_folder, category)
        n_images = len(glob(os.path.join(category_folder, '*[!_mask].png')))  # 확장자 .png로 수정
        print(f"{category}: {n_images}")