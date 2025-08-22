import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pickle
import json
import glob
from PIL import Image
from pathlib import Path

import sys
sys.path.append('/userHome/userhome2/donghee/modelcombination')

# Settings
iter = 8
title_ft = 24
score_ft = 24
save_dir = '/userHome/userhome2/donghee/modelcombination/result_paper/'
os.makedirs(save_dir,exist_ok=True)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_image_for_display(path, size=(352,352)):
    """시각화용 이미지 로드"""
    img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # 컬러 이미지로 변환
    if len(img_cv.shape) == 2:
        img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    else:
        img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv_color = cv2.resize(img_cv_color, size)
    
    return img_cv_color

def load_gt_for_display(path, size=(352,352)):
    """시각화용 Ground Truth 로드"""
    gt = Image.open(path).convert('L').resize(size)
    return np.array(gt) / 255.0

def load_test_outputs(output_dir):
    """테스트 출력 결과를 로드하는 함수 - npy 파일만 처리"""
    test_outputs = {}
    
    # .npy 파일들 찾기
    npy_files = glob.glob(os.path.join(output_dir, "*.npy"))
    
    if npy_files:
        print(f"  Found {len(npy_files)} .npy files")
        for npy_file in npy_files:
            # 파일명에서 확장자 제거 (예: cju123.png.npy -> cju123.png)
            img_name = Path(npy_file).stem
            try:
                pred = np.load(npy_file)
                
                # 예측 결과 형태 확인 및 변환
                original_shape = pred.shape
                
                # 4D인 경우: (1, 1, H, W) 또는 (N, C, H, W)
                if len(pred.shape) == 4:
                    pred = pred[0, 0]  # 첫 번째 배치, 첫 번째 채널
                # 3D인 경우: (1, H, W) 또는 (C, H, W) 또는 (H, W, 1)
                elif len(pred.shape) == 3:
                    if pred.shape[0] == 1:  # (1, H, W)
                        pred = pred[0]
                    elif pred.shape[2] == 1:  # (H, W, 1)
                        pred = pred[:, :, 0]
                    else:  # (C, H, W) where C > 1
                        pred = pred[0]  # 첫 번째 채널 사용
                
                # 값 범위 확인 및 정규화
                if pred.max() > 1.0:
                    pred = pred.astype(np.float32) / 255.0
                else:
                    pred = pred.astype(np.float32)
                
                test_outputs[img_name] = pred
                
                # 처음 3개 파일만 상세 정보 출력
                if len(test_outputs) <= 3:
                    print(f"    Loaded {img_name}: {original_shape} -> {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
                    
            except Exception as e:
                print(f"    Error loading {npy_file}: {e}")
    else:
        print(f"  No .npy files found in {output_dir}")
    
    return test_outputs

def dice(pred, gt):
    """Dice coefficient 계산 - 수정된 버전"""
    # 입력 검증
    if pred.shape != gt.shape:
        print(f"Warning: Shape mismatch - pred: {pred.shape}, gt: {gt.shape}")
        return 0.0
    
    # 이진화
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    # 교집합과 합집합 계산
    intersection = np.sum(pred_bin * gt_bin)
    pred_sum = np.sum(pred_bin)
    gt_sum = np.sum(gt_bin)
    
    # Dice 계수 = 2 * |A ∩ B| / (|A| + |B|)
    union = pred_sum + gt_sum
    
    if union == 0:
        # 둘 다 빈 마스크인 경우
        return 1.0
    
    dice_score = (2.0 * intersection) / union
    return dice_score

def calculate_polyp_size(gt_mask):
    """용종 크기 계산 - 전체 이미지 대비 용종 영역의 비율"""
    total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
    polyp_pixels = np.sum(gt_mask > 0.5)
    polyp_ratio = polyp_pixels / total_pixels
    return polyp_ratio, polyp_pixels

def categorize_polyp_size(polyp_ratio):
    """용종 크기를 카테고리로 분류"""
    if polyp_ratio < 0.05:  # 5% 미만
        return "Small"
    elif polyp_ratio < 0.15:  # 5-15%
        return "Medium"
    elif polyp_ratio < 0.30:  # 15-30%
        return "Large"
    else:  # 30% 이상
        return "Very Large"

def find_corresponding_file(pred_filename, base_dir, target_extensions=['.jpg', '.png', '.tif', '.jpeg']):
    """
    예측 파일명에 해당하는 원본 파일을 찾는 함수
    예: 'cju123.png' -> 'cju123.jpg' 찾기
    """
    # .npy에서 온 파일명이므로 원래 확장자가 남아있을 수 있음
    if '.' in pred_filename:
        base_name = pred_filename.rsplit('.', 1)[0]  # 확장자 제거
    else:
        base_name = pred_filename
    
    # 각 확장자로 시도
    for ext in target_extensions:
        potential_path = os.path.join(base_dir, f"{base_name}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    
    # 디버깅: 비슷한 이름의 파일들 찾기
    if os.path.exists(base_dir):
        all_files = os.listdir(base_dir)
        similar_files = [f for f in all_files if base_name in f]
        if similar_files:
            # 정확히 매칭되는 파일 찾기
            for f in similar_files:
                f_base = f.rsplit('.', 1)[0] if '.' in f else f
                if f_base == base_name:
                    return os.path.join(base_dir, f)
    
    return None

def analyze_large_polyp_performance(model_predictions, images_base_dir, masks_base_dir, focus_on_large=True):
    """
    큰 용종에서 One-skip의 성능 저하를 분석하는 함수
    focus_on_large=True: 큰 용종에서 One-skip이 다른 모델보다 성능이 떨어지는 경우를 찾음
    focus_on_large=False: 모든 용종 크기별 성능 분석
    """
    print(f"\n=== Analyzing One-skip performance on large polyps ===")
    
    # 모든 모델에 공통으로 있는 이미지들 찾기
    all_models = list(model_predictions.keys())
    if 'One-skip' not in all_models:
        print("Error: One-skip model not found!")
        return []
    
    # 모든 모델에 예측이 있는 이미지들만 고려
    common_images = set(model_predictions['One-skip'].keys())
    for model_name in all_models:
        if model_name != 'One-skip':
            common_images = common_images.intersection(set(model_predictions[model_name].keys()))
    
    print(f"Found {len(common_images)} images common to all models")
    
    if not common_images:
        print("No common images found across all models!")
        return []
    
    analyzed_images = []
    processed_count = 0
    error_count = 0
    
    # 용종 크기별 통계
    size_stats = {
        "Small": {"count": 0, "one_skip_best": 0, "one_skip_worst": 0},
        "Medium": {"count": 0, "one_skip_best": 0, "one_skip_worst": 0},
        "Large": {"count": 0, "one_skip_best": 0, "one_skip_worst": 0},
        "Very Large": {"count": 0, "one_skip_best": 0, "one_skip_worst": 0}
    }
    
    print(f"Analyzing polyp sizes and performance...")
    print(f"Looking for image files in: {images_base_dir}")
    print(f"Looking for mask files in: {masks_base_dir}")
    
    for i, img_name in enumerate(common_images):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(common_images)} images...")
        
        try:
            # 해당하는 원본 이미지와 마스크 파일 찾기
            img_path = find_corresponding_file(img_name, images_base_dir)
            mask_path = find_corresponding_file(img_name, masks_base_dir)
            
            if img_path is None or mask_path is None:
                if error_count < 5:
                    print(f"    Missing files for '{img_name}'")
                error_count += 1
                continue
            
            # Ground Truth 로드
            try:
                gt = load_gt_for_display(mask_path)
            except Exception as e:
                if error_count < 5:
                    print(f"    Error loading GT for '{img_name}': {e}")
                error_count += 1
                continue
            
            # 용종 크기 계산
            polyp_ratio, polyp_pixels = calculate_polyp_size(gt)
            polyp_size_category = categorize_polyp_size(polyp_ratio)
            
            # 모든 모델의 Dice 점수 계산
            all_scores = {}
            valid_predictions = True
            
            for model_name in all_models:
                try:
                    pred = model_predictions[model_name][img_name].copy()
                    
                    # 크기 조정
                    if pred.shape != gt.shape:
                        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                    
                    # 정규화
                    if pred.max() > 1.0:
                        pred = pred.astype(np.float32) / 255.0
                    
                    # Dice 점수 계산
                    score = dice(pred, gt)
                    all_scores[model_name] = score
                    
                except Exception as e:
                    if error_count < 5:
                        print(f"    Error processing {model_name} for '{img_name}': {e}")
                    valid_predictions = False
                    break
            
            if not valid_predictions:
                error_count += 1
                continue
            
            # One-skip 성능 분석
            one_skip_score = all_scores['One-skip']
            other_scores = [score for model, score in all_scores.items() if model != 'One-skip']
            max_other_score = max(other_scores)
            min_other_score = min(other_scores)
            
            # 성능 순위 계산
            all_score_pairs = [(model, score) for model, score in all_scores.items()]
            all_score_pairs.sort(key=lambda x: x[1], reverse=True)
            one_skip_rank = [i for i, (model, _) in enumerate(all_score_pairs) if model == 'One-skip'][0] + 1
            
            # 최고 성능 모델과 최저 성능 모델
            best_model = all_score_pairs[0][0]
            worst_model = all_score_pairs[-1][0]
            
            # 통계 업데이트
            size_stats[polyp_size_category]["count"] += 1
            if best_model == 'One-skip':
                size_stats[polyp_size_category]["one_skip_best"] += 1
            if worst_model == 'One-skip':
                size_stats[polyp_size_category]["one_skip_worst"] += 1
            
            # 성능 차이 계산
            performance_gap = one_skip_score - max_other_score  # 양수면 One-skip이 더 좋음
            
            analyzed_images.append({
                'img_name': img_name,
                'img_path': img_path,
                'mask_path': mask_path,
                'polyp_ratio': polyp_ratio,
                'polyp_pixels': polyp_pixels,
                'polyp_size_category': polyp_size_category,
                'one_skip_score': one_skip_score,
                'max_other_score': max_other_score,
                'performance_gap': performance_gap,
                'one_skip_rank': one_skip_rank,
                'best_model': best_model,
                'worst_model': worst_model,
                'all_scores': all_scores
            })
            
            processed_count += 1
            
        except Exception as e:
            if error_count < 5:
                print(f"    Unexpected error processing '{img_name}': {e}")
            error_count += 1
            continue
    
    print(f"\nProcessing summary:")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors encountered: {error_count}")
    
    # 용종 크기별 통계 출력
    print(f"\n=== Polyp Size Distribution and One-skip Performance ===")
    for size_category, stats in size_stats.items():
        count = stats["count"]
        if count > 0:
            best_rate = (stats["one_skip_best"] / count) * 100
            worst_rate = (stats["one_skip_worst"] / count) * 100
            print(f"{size_category:>10} polyps: {count:>3} images")
            print(f"           One-skip BEST:  {stats['one_skip_best']:>3}/{count} ({best_rate:>5.1f}%)")
            print(f"           One-skip WORST: {stats['one_skip_worst']:>3}/{count} ({worst_rate:>5.1f}%)")
    
    if not analyzed_images:
        print("No valid analyzed images found!")
        return []
    
    # 큰 용종에서 One-skip 성능이 떨어지는 경우 필터링
    if focus_on_large:
        # 큰 용종 (Large, Very Large)에서 One-skip이 다른 모델보다 성능이 떨어지는 경우
        large_polyp_poor_performance = [
            img for img in analyzed_images 
            if img['polyp_size_category'] in ['Large', 'Very Large'] 
            and img['performance_gap'] < 0  # One-skip이 다른 모델보다 성능이 떨어짐
        ]
        
        # 성능 차이가 큰 순으로 정렬 (One-skip이 가장 많이 떨어지는 순)
        large_polyp_poor_performance.sort(key=lambda x: x['performance_gap'])
        
        print(f"\n=== Large polyps where One-skip underperforms ===")
        print(f"Found {len(large_polyp_poor_performance)} large polyps where One-skip performs worse")
        
        selected_images = large_polyp_poor_performance[:15]  # 상위 15개
        
        for i, img_info in enumerate(selected_images):
            print(f"  {i+1}. {img_info['img_name']} ({img_info['polyp_size_category']}):")
            print(f"     Polyp size: {img_info['polyp_ratio']*100:.1f}% of image")
            print(f"     One-skip: {img_info['one_skip_score']:.4f} (rank #{img_info['one_skip_rank']})")
            print(f"     Best other: {img_info['max_other_score']:.4f} ({img_info['best_model']})")
            print(f"     Performance gap: {img_info['performance_gap']:.4f}")
        
        return selected_images
    
    else:
        # 모든 용종 크기별 분석
        return analyzed_images

def create_size_performance_plot(analyzed_images):
    """용종 크기별 성능 분석 플롯 생성"""
    
    # 용종 크기별 데이터 분리
    size_categories = ["Small", "Medium", "Large", "Very Large"]
    model_names = list(analyzed_images[0]['all_scores'].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance by Polyp Size', fontsize=16, fontweight='bold')
    
    for idx, size_cat in enumerate(size_categories):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 해당 크기의 용종들 필터링
        size_data = [img for img in analyzed_images if img['polyp_size_category'] == size_cat]
        
        if not size_data:
            ax.text(0.5, 0.5, f'No {size_cat} polyps found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{size_cat} Polyps (n=0)')
            continue
        
        # 각 모델의 평균 성능 계산
        model_scores = {}
        for model in model_names:
            scores = [img['all_scores'][model] for img in size_data]
            model_scores[model] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        # 바 플롯
        x_pos = np.arange(len(model_names))
        means = [model_scores[model]['mean'] for model in model_names]
        stds = [model_scores[model]['std'] for model in model_names]
        
        colors = ['red' if model == 'One-skip' else 'skyblue' for model in model_names]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        
        # One-skip 바 강조
        for i, model in enumerate(model_names):
            if model == 'One-skip':
                bars[i].set_edgecolor('darkred')
                bars[i].set_linewidth(2)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Dice Score')
        ax.set_title(f'{size_cat} Polyps (n={len(size_data)})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 평균값 텍스트 표시
        for i, (model, mean) in enumerate(zip(model_names, means)):
            ax.text(i, mean + stds[i] + 0.02, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir+"polyp_size_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved polyp_size_performance_analysis.png")

# Config - 각 모델의 test_outputs 디렉토리 경로 설정
models_config = {
    'One-skip': {
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250630_202335/crc_test_v72_Iter_{iter}/test_outputs',
        'display_name': 'One-skip'
    },
    'Two-skip': {
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250627_212507/crc_test_v71_Iter_{iter}/test_outputs',
        'display_name': 'Two-skip'
    },
    'Three-skip': {
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250702_152021/crc_test_v73_Iter_{iter}/test_outputs',
        'display_name': 'Three-skip'
    },
    'Four-skip': {
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250704_221830/crc_test_v74_Iter_{iter}/test_outputs',
        'display_name': 'Four-skip'
    }
}

# 원본 이미지와 GT 경로
images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/images"
masks_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/masks"

# 각 모델의 test outputs 로드
print("=== Loading Test Outputs ===")
model_predictions = {}

for model_name, config in models_config.items():
    output_dir = config['test_outputs_dir']
    print(f"\nLoading {model_name} from: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"  Warning: Directory does not exist: {output_dir}")
        model_predictions[model_name] = {}
        continue
    
    try:
        predictions = load_test_outputs(output_dir)
        model_predictions[model_name] = predictions
        print(f"  Loaded {len(predictions)} predictions")
        
        # 사용 가능한 이미지 이름들 출력 (처음 5개만)
        available_images = list(predictions.keys())[:5]
        print(f"  Sample images: {available_images}")
        
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        model_predictions[model_name] = {}

# 큰 용종에서 One-skip 성능 저하 분석
if 'One-skip' not in model_predictions or len(model_predictions['One-skip']) == 0:
    print("Error: One-skip model predictions not found!")
    exit()

# 먼저 전체 분석 수행
all_analyzed_images = analyze_large_polyp_performance(
    model_predictions, 
    images_base_dir,
    masks_base_dir, 
    focus_on_large=False
)

if all_analyzed_images:
    # 용종 크기별 성능 분석 플롯 생성
    create_size_performance_plot(all_analyzed_images)

# 큰 용종에서 One-skip 성능 저하 사례 찾기
selected_images = analyze_large_polyp_performance(
    model_predictions, 
    images_base_dir,
    masks_base_dir, 
    focus_on_large=True  # 큰 용종에서 성능 저하 사례만
)

if not selected_images:
    print("No large polyps found where One-skip underperforms!")
    exit()

# 결과 처리 및 시각화
print("\n=== Processing Selected Results ===")
results = []

for img_info in selected_images:
    img_name = img_info['img_name']
    print(f"\nProcessing {img_name}")
    
    try:
        # 이미 찾은 경로 사용
        img_path = img_info['img_path']
        mask_path = img_info['mask_path']
        
        img_np = load_image_for_display(img_path)
        gt = load_gt_for_display(mask_path)
        
    except Exception as e:
        print(f"  Error loading files: {e}")
        continue
    
    # 각 모델의 예측 결과 준비
    preds = {}
    dscs = img_info['all_scores']  # 이미 계산된 Dice 점수 사용
    
    for model_name in models_config.keys():
        try:
            if model_name in model_predictions and img_name in model_predictions[model_name]:
                pred = model_predictions[model_name][img_name].copy()
                
                # 예측 결과 크기 조정
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                
                # 값 범위 정규화
                if pred.max() > 1.0:
                    pred = pred.astype(np.float32) / 255.0
                
                preds[model_name] = pred
                print(f"  {model_name}: DSC={dscs[model_name]:.4f}")
            else:
                preds[model_name] = np.zeros_like(gt)
                if model_name not in dscs:
                    dscs[model_name] = 0.0
                print(f"  {model_name}: No prediction available")
                
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            preds[model_name] = np.zeros_like(gt)
            dscs[model_name] = 0.0
    
    results.append({
        'img_name': img_name,
        'img': img_np, 
        'gt': gt,
        'preds': preds, 
        'dscs': dscs,
        'polyp_size_category': img_info['polyp_size_category'],
        'polyp_ratio': img_info['polyp_ratio'],
        'performance_gap': img_info['performance_gap'],
        'best_model': img_info['best_model']
    })

# Visualization
print("\n=== Creating Visualization ===")
if not results:
    print("No results to visualize!")
    exit()

n_models = len(models_config) + 2
fig, axes = plt.subplots(len(results), n_models, figsize=(3*n_models, 6*len(results)))

if len(results) == 1:
    axes = np.expand_dims(axes, 0)

# 첫 번째 행에만 컬럼 제목 추가
column_titles = ["Input", "Ground Truth"] + [config['display_name'] for config in models_config.values()]


for col, title in enumerate(column_titles):
    if title == "One-skip":  # One-skip 강조 (성능 저하 사례이므로 다른 색상)
        axes[0, col].text(0.5, 1.15, title, 
                         transform=axes[0, col].transAxes,
                         ha='center', va='center', 
                         fontsize=title_ft, fontweight='bold', color='darkred')
    else:
        axes[0, col].text(0.5, 1.15, title, 
                         transform=axes[0, col].transAxes,
                         ha='center', va='center', 
                         fontsize=title_ft, fontweight='bold')

for row, res in enumerate(results):
    # Input 이미지
    axes[row, 0].imshow(res['img'])
    axes[row, 0].axis('off')
    
    # 이미지 이름과 용종 정보 표시
    display_name = res['img_name']
    # if '.' in display_name:
    #     display_name = display_name.rsplit('.', 1)[0]
    
    # axes[row, 0].text(0.5, -0.08, f"{display_name}\n{res['polyp_size_category']}\nSize: {res['polyp_ratio']*100:.1f}%", 
    #                  transform=axes[row, 0].transAxes,
    #                  ha='center', va='top', 
    #                  fontsize=score_ft, fontweight='bold')

    # Ground Truth
    axes[row, 1].imshow(res['gt'], cmap='gray')
    axes[row, 1].axis('off')

    # 모델 예측 결과들
    for col, (model_name, pred) in enumerate(res['preds'].items()):
        axes[row, col+2].imshow(pred, cmap='gray')
        axes[row, col+2].axis('off')
        
        # DSC 점수 표시
        if model_name == 'One-skip':
            # One-skip은 성능이 떨어지는 사례이므로 빨간색으로 강조
            axes[row, col+2].text(0.5, -0.08, f"DSC: {res['dscs'][model_name]:.4f}", 
                                 transform=axes[row, col+2].transAxes,
                                 ha='center', va='top', 
                                 fontsize=score_ft, fontweight='bold', color='darkred')
        elif model_name == res['best_model']:
            # 최고 성능 모델은 녹색으로 표시
            axes[row, col+2].text(0.5, -0.08, f"DSC: {res['dscs'][model_name]:.4f}", 
                                 transform=axes[row, col+2].transAxes,
                                 ha='center', va='top', 
                                 fontsize=score_ft, fontweight='bold', color='darkgreen')
        else:
            axes[row, col+2].text(0.5, -0.08, f"DSC: {res['dscs'][model_name]:.4f}", 
                                 transform=axes[row, col+2].transAxes,
                                 ha='center', va='top', 
                                 fontsize=score_ft, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig(save_dir+"large_polyp_one_skip_underperformance.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved large_polyp_one_skip_underperformance.png")

# 성능 통계 출력 (큰 용종에서 One-skip 성능 저하 사례들)
print("\n=== Performance Statistics for Large Polyps where One-skip Underperforms ===")
for model_name in models_config.keys():
    model_dsc_scores = [res['dscs'][model_name] for res in results]
    mean_dsc = np.mean(model_dsc_scores)
    std_dsc = np.std(model_dsc_scores)
    if model_name == 'One-skip':
        print(f"{model_name}: Mean DSC = {mean_dsc:.4f} ± {std_dsc:.4f} ❌ (UNDERPERFORMING)")
    else:
        print(f"{model_name}: Mean DSC = {mean_dsc:.4f} ± {std_dsc:.4f}")

# 용종 크기별 추가 분석
print(f"\n=== Detailed Analysis of Large Polyp Cases ===")
large_cases = [res for res in results if res['polyp_size_category'] in ['Large', 'Very Large']]
if large_cases:
    print(f"Large polyp cases analyzed: {len(large_cases)}")
    
    # 용종 크기 분포
    size_distribution = {}
    for res in large_cases:
        size_cat = res['polyp_size_category']
        if size_cat not in size_distribution:
            size_distribution[size_cat] = []
        size_distribution[size_cat].append(res['polyp_ratio'])
    
    for size_cat, ratios in size_distribution.items():
        print(f"{size_cat} polyps: {len(ratios)} cases")
        print(f"  Size range: {min(ratios)*100:.1f}% - {max(ratios)*100:.1f}% of image")
        print(f"  Average size: {np.mean(ratios)*100:.1f}% of image")
    
    # One-skip과 최고 성능 모델 비교
    one_skip_scores = [res['dscs']['One-skip'] for res in large_cases]
    best_model_scores = [res['dscs'][res['best_model']] for res in large_cases]
    performance_gaps = [res['performance_gap'] for res in large_cases]
    
    print(f"\nPerformance Gap Analysis:")
    print(f"  Average One-skip score: {np.mean(one_skip_scores):.4f}")
    print(f"  Average best model score: {np.mean(best_model_scores):.4f}")
    print(f"  Average performance gap: {np.mean(performance_gaps):.4f}")
    print(f"  Largest performance gap: {min(performance_gaps):.4f}")
    
    # 최고 성능 모델 분포
    best_models = [res['best_model'] for res in large_cases]
    from collections import Counter
    best_model_counts = Counter(best_models)
    print(f"\nBest performing models on large polyps:")
    for model, count in best_model_counts.most_common():
        print(f"  {model}: {count}/{len(large_cases)} cases ({count/len(large_cases)*100:.1f}%)")

print(f"\nProcessed {len(results)} large polyp cases where One-skip underperforms!")
print("These cases show One-skip struggling with larger polyps")
print("Check large_polyp_one_skip_underperformance.png for visual comparison")
print("Check polyp_size_performance_analysis.png for overall size-based performance analysis")