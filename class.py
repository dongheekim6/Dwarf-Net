import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict

def calculate_mask_ratio(mask_path):
    """마스크 이미지에서 폴리프가 차지하는 비율을 계산"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    
    # 마스크에서 폴리프 영역(흰색 픽셀)의 비율 계산
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = np.sum(mask == 255)
    ratio = (white_pixels / total_pixels) * 100
    
    return ratio

def classify_masks(images_path, masks_path, output_base_path, file_extension):
    """마스크 이미지를 비율에 따라 분류하고 해당하는 원본 이미지도 함께 복사"""
    # 결과 저장할 디렉토리 생성
    thresholds = [25, 10, 5, 2]  # 임계값 수정
    result_dirs = {}
    
    for threshold in thresholds:
        result_dir = os.path.join(output_base_path, f"below_{threshold}percent")
        # images와 masks 서브디렉토리 생성
        os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "masks"), exist_ok=True)
        result_dirs[threshold] = result_dir

    # 마스크 이미지 처리 - 데이터셋별 파일 확장자 처리
    mask_files = list(Path(masks_path).glob(f'*.{file_extension}'))
    results = {threshold: [] for threshold in thresholds}
    
    print(f"\n데이터셋 처리 중... 총 {len(mask_files)}개 이미지")
    
    for mask_path in tqdm(mask_files, desc="이미지 처리"):
        mask_filename = mask_path.name
        ratio = calculate_mask_ratio(str(mask_path))
        
        # 원본 이미지 경로
        original_path = Path(images_path) / mask_filename
        
        if not original_path.exists():
            print(f"경고: {mask_filename}에 대한 원본 이미지를 찾을 수 없습니다.")
            continue
        
        # 각 임계값에 따라 분류
        for threshold in thresholds:
            if ratio <= threshold:
                results[threshold].append(mask_path)
                
                # 결과 디렉토리
                dest_dir = result_dirs[threshold]
                
                # 마스크 이미지 복사 - masks 폴더로
                mask_dest = os.path.join(dest_dir, "masks", mask_filename)
                shutil.copy2(mask_path, mask_dest)
                
                # 원본 이미지 복사 - images 폴더로
                original_dest = os.path.join(dest_dir, "images", mask_filename)
                shutil.copy2(original_path, original_dest)
                
                # 복사 확인 메시지
                print(f"\n{threshold}% 이하 그룹에 추가됨: {mask_filename}")
                print(f"  원본 이미지: {original_dest}")
                print(f"  마스크 이미지: {mask_dest}")

    return results, len(mask_files)

def process_all_datasets():
    """모든 데이터셋 처리"""
    datasets = {
        'CVC-ColonDB': {
            'images_path': '/userHome/userhome2/donghee/modelcombination/CVC-ColonDB/images',
            'masks_path': '/userHome/userhome2/donghee/modelcombination/CVC-ColonDB/masks',
            'extension': 'png'
        },
        'Kvasir-SEG': {
            'images_path': '/userHome/userhome2/donghee/modelcombination/Kvasir-SEG/images',
            'masks_path': '/userHome/userhome2/donghee/modelcombination/Kvasir-SEG/masks',
            'extension': 'jpg'
        },
        'CVC-ClinicDB': {
            'images_path': '/userHome/userhome2/donghee/modelcombination/CVC-ClinicDB/images',
            'masks_path': '/userHome/userhome2/donghee/modelcombination/CVC-ClinicDB/masks',
            'extension': 'tif'
        },
        'ETIS': {
            'images_path': '/userHome/userhome2/donghee/modelcombination/ETIS-LaribPolypDB/images',
            'masks_path': '/userHome/userhome2/donghee/modelcombination/ETIS-LaribPolypDB/masks',
            'extension': 'png'
        }
    }
    
    # 전체 통계를 위한 변수들
    total_stats = defaultdict(int)
    dataset_total_images = {}
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{dataset_name} 데이터셋 처리 시작...")
        output_path = f"./results/{dataset_name}"
        results, total_images = classify_masks(
            dataset_info['images_path'],
            dataset_info['masks_path'], 
            output_path, 
            dataset_info['extension']
        )
        
        # 데이터셋별 총 이미지 수 저장
        dataset_total_images[dataset_name] = total_images
        
        # 결과 출력 및 전체 통계 업데이트
        print(f"\n{dataset_name} 처리 결과:")
        for threshold, files in results.items():
            count = len(files)
            total_stats[threshold] += count
            percentage = (count / total_images) * 100
            print(f"{threshold}% 이하 이미지 수: {count} ({percentage:.1f}%)")
    
    # 전체 통계 출력
    print("\n=== 전체 데이터셋 통계 ===")
    total_all_images = sum(dataset_total_images.values())
    print(f"전체 이미지 수: {total_all_images}")
    
    print("\n각 데이터셋 크기:")
    for dataset_name, total in dataset_total_images.items():
        percentage = (total / total_all_images) * 100
        print(f"{dataset_name}: {total} 이미지 ({percentage:.1f}%)")
    
    print("\n임계값별 전체 통계:")
    for threshold in sorted(total_stats.keys(), reverse=True):
        count = total_stats[threshold]
        percentage = (count / total_all_images) * 100
        print(f"{threshold}% 이하: {count} 이미지 ({percentage:.1f}%)")
            
if __name__ == "__main__":
    process_all_datasets()