import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from PIL import Image
import os

def analyze_test_results(pred_dir, gt_dir, output_dir):
    results = []
    pred_files = sorted(Path(pred_dir).glob('*.png.npy'))
    
    for pred_path in pred_files:
        case_num = pred_path.name.split('.')[0]
        gt_path = Path(gt_dir) / f"{case_num}.tif"
        
        if gt_path.exists():
            # 예측값 로드 및 전처리
            pred = np.load(pred_path)
            if pred.ndim == 3 and pred.shape[0] == 1:  # 첫 차원이 1인 경우
                pred = pred.squeeze(0)  # 첫 번째 차원 제거
            
            # Ground Truth 로드
            gt = np.array(Image.open(gt_path))
            
            # shape 확인 및 출력 (디버깅용)
            print(f"Case {case_num}:")
            print(f"Prediction shape: {pred.shape}")
            print(f"Ground truth shape: {gt.shape}")
            
            # shape이 다르면 transpose
            if gt.shape != pred.shape:
                gt = gt.transpose()
                print(f"After transpose - Ground truth shape: {gt.shape}")
            
            pred = (pred > 0.5).astype(np.uint8)
            gt = (gt > 0).astype(np.uint8)
            
            dice = calculate_dice(pred, gt)
            iou = calculate_iou(pred, gt)
            tumor_size = np.sum(gt)
            boundary_clarity = calculate_boundary_clarity(gt)
            
            results.append({
                'case_id': case_num,
                'dice': dice,
                'iou': iou,
                'tumor_size': tumor_size,
                'boundary_clarity': boundary_clarity
            })
    
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    
    # 성능 분포 시각화
    plt.figure(figsize=(15, 10))
    
    # Dice Score 분포
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='dice', bins=30)
    plt.title('Distribution of Dice Scores')
    
    # 분포의 주요 통계값 계산
    quantiles = df['dice'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for q, v in quantiles.items():
        plt.axvline(x=v, color='r', linestyle='--', alpha=0.5)
        plt.text(v, plt.ylim()[1]*0.9, f'{int(q*100)}%: {v:.3f}', rotation=90)
    
    # Tumor Size vs Dice Score
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='tumor_size', y='dice')
    plt.title('Tumor Size vs Dice Score')
    
    # Boundary Clarity vs Dice Score
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='boundary_clarity', y='dice')
    plt.title('Boundary Clarity vs Dice Score')
    
    # Box plot for Dice scores
    plt.subplot(2, 2, 4)
    sns.boxplot(y=df['dice'])
    plt.title('Dice Score Box Plot')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_analysis.png')
    
    # 기본 통계 정보 출력
    print("\n=== Dice Score 분포 통계 ===")
    print(df['dice'].describe())
    print("\n=== 백분위수 ===")
    print(quantiles)
    
    return df

def calculate_dice(pred, gt):
    """Dice coefficient 계산"""
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)

def calculate_iou(pred, gt):
    """IoU 계산"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-6)

def calculate_boundary_clarity(mask):
    """경계선 선명도 계산"""
    from scipy import ndimage
    gradient = ndimage.sobel(mask)
    return np.mean(gradient)

def analyze_specific_range(df, threshold):
    """특정 임계값 이하의 케이스들을 분석"""
    poor_cases = df[df['dice'] < threshold].sort_values('dice')
    print(f"\n=== Dice Score가 {threshold:.3f} 미만인 케이스들 ===")
    print(f"케이스 수: {len(poor_cases)}")
    print("\n상위 10개 케이스:")
    print(poor_cases[['case_id', 'dice', 'tumor_size', 'boundary_clarity']].head(10))
    return poor_cases

if __name__ == "__main__":
    pred_dir = '/userHome/userhome2/donghee/modelcombination/output_DCSA_UNet_cvc_noaug/output_250204_024844/DCSA_UNet_Iter_9/test_outputs'
    gt_dir = '/userHome/userhome2/donghee/modelcombination/masks'
    output_dir = './analysis_results'
    
    # 전체 분포 분석
    df = analyze_test_results(pred_dir, gt_dir, output_dir)
    
    # 분포를 보고 사용자가 임계값을 정할 수 있음
    # 예: 하위 25% 지점의 값을 임계값으로 사용
    threshold = df['dice'].quantile(0.25)
    poor_cases = analyze_specific_range(df, threshold)