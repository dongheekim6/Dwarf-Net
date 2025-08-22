import os
import cv2
import numpy as np

def count_mask_images_by_ratio(mask_dir, target_size=(352, 352)):
    """
    다양한 확장자의 마스크 이미지들 중, 전체 픽셀 대비 비율에 따라 분류하여 이미지 수를 출력합니다.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(valid_extensions)]
    
    total_images = 0
    count_0_5 = 0    # 0-5%
    count_5_10 = 0   # 5-10%
    count_10_25 = 0  # 10-25%
    count_25_50 = 0  # 25-50%
    count_50_100 = 0 # 50% 이상
    
    total_pixels = target_size[0] * target_size[1]
    
    for file in mask_files:
        path = os.path.join(mask_dir, file)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ 실패: {file} 로드하지 못함")
            continue
        
        if mask.shape != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
        mask_bin = (mask > 0).astype(np.uint8)
        mask_pixels = np.sum(mask_bin)
        ratio = (mask_pixels / total_pixels) * 100
        
        total_images += 1
        
        # 각 범위별로 카운트
        if ratio < 5:
            count_0_5 += 1
        elif ratio < 10:
            count_5_10 += 1
        elif ratio < 25:
            count_10_25 += 1
        elif ratio < 50:
            count_25_50 += 1
        else:  # ratio >= 50
            count_50_100 += 1
    
    print(f"\n✅ 총 마스크 이미지 수: {total_images}")
    print("\n📊 비율별 이미지 분포:")
    print(f"  0~5%:    {count_0_5:3d}장 ({count_0_5/total_images*100:5.1f}%)")
    print(f"  5~10%:   {count_5_10:3d}장 ({count_5_10/total_images*100:5.1f}%)")
    print(f"  10~25%:  {count_10_25:3d}장 ({count_10_25/total_images*100:5.1f}%)")
    print(f"  25~50%:  {count_25_50:3d}장 ({count_25_50/total_images*100:5.1f}%)")
    print(f"  50%이상: {count_50_100:3d}장 ({count_50_100/total_images*100:5.1f}%)")

# 실행
mask_folder = input("📂 마스크 이미지 폴더 경로를 입력하세요: ")
count_mask_images_by_ratio(mask_folder)