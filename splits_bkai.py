import os
import shutil
import random
from pathlib import Path
import numpy as np
from collections import defaultdict
import cv2
def create_kvasir_splits(dataset_path, output_path, num_splits=10, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Kvasir 데이터셋을 여러 개의 분할로 나누는 함수
    
    Args:
        dataset_path (str): 원본 Kvasir 데이터셋 경로 (images, masks 폴더가 있는 경로)
        output_path (str): 출력 경로 (splits 폴더가 생성될 경로)
        num_splits (int): 분할 개수 (기본값: 10)
        train_ratio (float): 훈련 데이터 비율 (기본값: 0.8)
        val_ratio (float): 검증 데이터 비율 (기본값: 0.1)
        test_ratio (float): 테스트 데이터 비율 (기본값: 0.1)
    """
    
    # 경로 설정
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    images_path = dataset_path / "train/train"
    masks_path = dataset_path / "train_gt/train_gt"
    splits_path = output_path / "splits"
    
    # 입력 데이터 검증
    if not images_path.exists():
        raise FileNotFoundError(f"Images 폴더를 찾을 수 없습니다: {images_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks 폴더를 찾을 수 없습니다: {masks_path}")
    
    # 이미지 파일 목록 가져오기 (jpg 파일만)
    image_files = list(images_path.glob("*.jpeg"))
    if not image_files:
        raise ValueError("images 폴더에 jpg 파일이 없습니다.")
    
    # 파일명에서 확장자 제거하여 base name 추출
    base_names = [img.stem for img in image_files]
    
    # 해당하는 마스크 파일이 있는지 확인
    valid_pairs = []
    for base_name in base_names:
        mask_file = masks_path / f"{base_name}.jpeg"
        if mask_file.exists():
            valid_pairs.append(base_name)
        else:
            print(f"경고: {base_name}에 해당하는 마스크 파일이 없습니다.")
    
    if not valid_pairs:
        raise ValueError("유효한 이미지-마스크 쌍이 없습니다.")
    
    print(f"총 {len(valid_pairs)}개의 유효한 이미지-마스크 쌍을 찾았습니다.")
    
    # 데이터셋 크기 계산
    total_samples = len(valid_pairs)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    print(f"분할 비율: Training={train_size}, Validation={val_size}, Test={test_size}")
    
    # splits 폴더 생성
    splits_path.mkdir(parents=True, exist_ok=True)
    
    # 각 분할에 대해 처리
    for split_idx in range(1, num_splits + 1):
        print(f"\nSplit {split_idx:02d} 생성 중...")
        
        # 현재 분할 폴더 경로
        current_split_path = splits_path / f"split{split_idx:02d}"
        
        # 하위 폴더 생성
        for subset in ["training", "validation", "test"]:
            subset_path = current_split_path / subset
            subset_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터를 랜덤하게 섞기 (매번 다른 시드 사용)
        random.seed(split_idx * 42)  # 각 split마다 다른 시드 사용
        shuffled_pairs = valid_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # 데이터 분할 (중복 없이)
        train_pairs = shuffled_pairs[:train_size]
        val_pairs = shuffled_pairs[train_size:train_size + val_size]
        test_pairs = shuffled_pairs[train_size + val_size:]
        
        # 중복 검사 (각 split 내에서)
        train_set = set(train_pairs)
        val_set = set(val_pairs)
        test_set = set(test_pairs)
        
        # 교집합 확인
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"경고: Split {split_idx:02d}에서 서브셋 간 중복 발견!")
            if train_val_overlap:
                print(f"  Training-Validation 중복: {len(train_val_overlap)}개")
            if train_test_overlap:
                print(f"  Training-Test 중복: {len(train_test_overlap)}개")
            if val_test_overlap:
                print(f"  Validation-Test 중복: {len(val_test_overlap)}개")
        
        # 파일 복사 함수
        def copy_files(pairs, subset_name):
            subset_path = current_split_path / subset_name
            for base_name in pairs:
                # 이미지 파일 복사
                src_img = images_path / f"{base_name}.jpeg"
                dst_img = subset_path / f"{base_name}.jpg"
                shutil.copy2(src_img, dst_img)
                
                # 마스크 파일 복사 (마스크는 _mask 접미사 추가)
                src_mask = masks_path / f"{base_name}.jpeg"
                dst_mask = subset_path / f"{base_name}_mask.jpg"
                shutil.copy2(src_mask, dst_mask)
                mask = cv2.imread(str(dst_mask), cv2.IMREAD_GRAYSCALE)
                #print(np.unique(mask))

                max_val = mask.max()
                mask_scaled = (mask.astype(np.float32) / max_val) * 255
                mask_scaled = mask_scaled.astype(np.uint8)

                _, binary_mask = cv2.threshold(mask_scaled, 50, 255, cv2.THRESH_BINARY)
                
                # mask_rgb = cv2.imread(dst_mask)
                # height,width = mask_rgb.shape[:2]
                # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
                # class_mask = np.zeros((height, width), dtype=np.uint8)
                # class_mask[np.all(mask_rgb == [0,0,0], axis=-1)] = 0
                # class_mask[np.all(mask_rgb == [0,255,0], axis=-1)] = 255
                # class_mask[np.all(mask_rgb == [255,0,0], axis=-1)] = 255
                
                binary_mask2 = np.where(mask > 0, 255, 0).astype(np.uint8)

                # 결과 저장 (선택)
                cv2.imwrite(str(dst_mask), binary_mask)
                #cv2.imwrite(str(dst_mask).replace('mask','mask_org'), binary_mask2)

        
        # 각 서브셋에 파일 복사
        copy_files(train_pairs, "training")
        copy_files(val_pairs, "validation")
        copy_files(test_pairs, "test")
        
        print(f"Split {split_idx:02d} 완료: Training={len(train_pairs)}, Validation={len(val_pairs)}, Test={len(test_pairs)}")
    
    print(f"\n모든 분할이 완료되었습니다. 결과는 {splits_path}에 저장되었습니다.")
    
    # 결과 요약 출력
    print("\n=== 분할 결과 요약 ===")
    for split_idx in range(1, num_splits + 1):
        split_path = splits_path / f"split{split_idx:02d}"
        train_count = len([f for f in (split_path / "training").glob("*.jpg") if not f.name.endswith("_mask.jpg")])
        val_count = len([f for f in (split_path / "validation").glob("*.jpg") if not f.name.endswith("_mask.jpg")])
        test_count = len([f for f in (split_path / "test").glob("*.jpg") if not f.name.endswith("_mask.jpg")])
        print(f"Split {split_idx:02d}: Training={train_count}, Validation={val_count}, Test={test_count}")


def calculate_overlap_statistics(splits_path, num_splits=10):
    """
    분할 간 중복률을 계산하는 함수
    """
    print("\n=== 중복률 계산 중 ===")
    splits_path = Path(splits_path) / "splits"
    
    # 각 분할의 파일 수집
    split_data = {}
    for split_idx in range(1, num_splits + 1):
        split_path = splits_path / f"split{split_idx:02d}"
        split_data[split_idx] = {
            'training': set(),
            'validation': set(),
            'test': set()
        }
        
        for subset in ["training", "validation", "test"]:
            subset_path = split_path / subset
            if subset_path.exists():
                files = set(f.stem for f in subset_path.glob("*.jpg") if not f.name.endswith("_mask.jpg"))
                split_data[split_idx][subset] = files
    
    # 1. 각 split 내부의 중복 검사
    print("\n1. 각 Split 내부 중복 검사:")
    for split_idx in range(1, num_splits + 1):
        train_set = split_data[split_idx]['training']
        val_set = split_data[split_idx]['validation']
        test_set = split_data[split_idx]['test']
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        total_files = len(train_set) + len(val_set) + len(test_set)
        
        print(f"Split {split_idx:02d}:")
        print(f"  Training-Validation 중복: {len(train_val_overlap)}개")
        print(f"  Training-Test 중복: {len(train_test_overlap)}개")
        print(f"  Validation-Test 중복: {len(val_test_overlap)}개")
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"  ⚠️  내부 중복 발견!")
        else:
            print(f"  ✅ 내부 중복 없음")
    
    # 2. Split 간 중복률 계산
    print("\n2. Split 간 중복률 계산:")
    
    # 같은 subset끼리 비교
    for subset in ["training", "validation", "test"]:
        print(f"\n{subset.upper()} 세트 간 중복률:")
        overlap_matrix = np.zeros((num_splits, num_splits))
        
        for i in range(num_splits):
            for j in range(i + 1, num_splits):
                split_i = i + 1
                split_j = j + 1
                
                set_i = split_data[split_i][subset]
                set_j = split_data[split_j][subset]
                
                overlap = set_i & set_j
                union = set_i | set_j
                
                # Jaccard 계수 (교집합/합집합)
                jaccard = len(overlap) / len(union) if len(union) > 0 else 0
                overlap_matrix[i][j] = jaccard
                overlap_matrix[j][i] = jaccard
                
                print(f"  Split {split_i:02d} vs Split {split_j:02d}: {len(overlap):3d}개 중복, Jaccard={jaccard:.3f}")
    
    # 3. 전체 데이터 사용률 계산
    print("\n3. 전체 데이터 사용률:")
    all_files_by_subset = defaultdict(set)
    
    for split_idx in range(1, num_splits + 1):
        for subset in ["training", "validation", "test"]:
            all_files_by_subset[subset].update(split_data[split_idx][subset])
    
    # 전체 고유 파일 수
    all_unique_files = set()
    for subset_files in all_files_by_subset.values():
        all_unique_files.update(subset_files)
    
    print(f"전체 고유 파일 수: {len(all_unique_files)}")
    for subset in ["training", "validation", "test"]:
        print(f"{subset.upper()} 세트에서 사용된 고유 파일 수: {len(all_files_by_subset[subset])}")
    
    # 4. 평균 중복률 계산
    print("\n4. 평균 중복률 요약:")
    for subset in ["training", "validation", "test"]:
        total_overlap = 0
        total_comparisons = 0
        
        for i in range(num_splits):
            for j in range(i + 1, num_splits):
                split_i = i + 1
                split_j = j + 1
                
                set_i = split_data[split_i][subset]
                set_j = split_data[split_j][subset]
                
                overlap = len(set_i & set_j)
                total_overlap += overlap
                total_comparisons += 1
        
        avg_overlap = total_overlap / total_comparisons if total_comparisons > 0 else 0
        print(f"{subset.upper()}: 평균 {avg_overlap:.1f}개 중복 (총 {total_comparisons}개 비교)")


def verify_splits(splits_path, num_splits=10):
    """
    생성된 분할의 무결성을 검증하는 함수
    """
    print("\n=== 분할 검증 중 ===")
    splits_path = Path(splits_path) / "splits"
    
    all_files_across_splits = set()
    
    for split_idx in range(1, num_splits + 1):
        split_path = splits_path / f"split{split_idx:02d}"
        
        # 각 분할 내에서 중복 확인
        split_files = set()
        for subset in ["training", "validation", "test"]:
            subset_path = split_path / subset
            if not subset_path.exists():
                print(f"경고: {subset_path} 폴더가 존재하지 않습니다.")
                continue
                
            subset_files = set(f.stem for f in subset_path.glob("*.jpg") if not f.name.endswith("_mask.jpg"))
            
            # 같은 분할 내에서 중복 확인
            overlap = split_files.intersection(subset_files)
            if overlap:
                print(f"❌ Split {split_idx:02d}에서 {subset}과 다른 서브셋 간에 중복된 파일이 있습니다: {overlap}")
            
            split_files.update(subset_files)
            
            # 이미지와 마스크 쌍 확인
            mask_files = set(f.stem.replace("_mask", "") for f in subset_path.glob("*_mask.jpg"))
            if subset_files != mask_files:
                missing_masks = subset_files - mask_files
                missing_images = mask_files - subset_files
                if missing_masks:
                    print(f"❌ Split {split_idx:02d}의 {subset}에서 마스크가 없는 이미지: {missing_masks}")
                if missing_images:
                    print(f"❌ Split {split_idx:02d}의 {subset}에서 이미지가 없는 마스크: {missing_images}")
            else:
                print(f"✅ Split {split_idx:02d}의 {subset}: 이미지-마스크 쌍 완벽 매칭")
        
        all_files_across_splits.update(split_files)
        print(f"Split {split_idx:02d}: {len(split_files)}개 파일 확인됨")
    
    print(f"총 {len(all_files_across_splits)}개의 고유한 파일이 모든 분할에 사용되었습니다.")
    print("✅ 검증 완료!")


# 사용 예시
if __name__ == "__main__":
    # 데이터셋 경로와 출력 경로 설정
    dataset_path = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/bkai-igh-neopolyp"  # Kvasir 데이터셋 경로 (images, masks 폴더가 있는 경로)
    output_path = "/userHome/userhome2/donghee/modelcombination/splits_bkai"  # 출력 경로
    os.makedirs(output_path, exist_ok=True)
    try:
        # 분할 생성
        create_kvasir_splits(
            dataset_path=dataset_path,
            output_path=output_path,
            num_splits=10,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # 분할 검증
        verify_splits(output_path, num_splits=10)
        
        # 중복률 계산
        calculate_overlap_statistics(output_path, num_splits=10)
        
    except Exception as e:
        print(f"오류 발생: {e}")