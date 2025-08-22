import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from PIL import Image
import os
import matplotlib.font_manager as fm

# 폰트 리스트 확인(필요 없으면 주석 처리 가능)
font_list = [f.name for f in fm.fontManager.ttflist]
print("Available fonts:", font_list)

def analyze_test_results(pred_dir, gt_dir, output_dir):
    """
    pred_dir 내 '*.jpg.npy' 형태의 예측 마스크와,
    동일한 파일명(확장자 .jpg)으로 구성된 GT 마스크를 로드하여
    (352, 352) 크기로 Dice, IoU 등을 분석하고 결과를 시각화합니다.
    """
    # 최종 결과 저장 폴더 (analysis_results/comparisons_dcsau_kvasir)
    save_root = os.path.join(output_dir, 'comparisons_emcad_kvasir_iter10')
    os.makedirs(save_root, exist_ok=True)

    results = []
    # 1) 예측 파일 패턴: '*.jpg.npy'
    pred_files = sorted(Path(pred_dir).glob('*.jpg.npy'))
    
    print(f"Found prediction files: {len(pred_files)}")

    for pred_path in pred_files:
        # 예: pred_path = "cju0sxqiclckk08551ycbwhno.jpg.npy"라면,
        # pred_path.stem -> "cju0sxqiclckk08551ycbwhno.jpg"
        base_stem = pred_path.stem  # "cju0sxqiclckk08551ycbwhno.jpg"
        
        # GT 파일 이름도 동일하게 "cju0sxqiclckk08551ycbwhno.jpg" 형식
        gt_filename = base_stem  # 그대로 사용
        gt_path = Path(gt_dir) / gt_filename
        
        print(f"\n[Processing]")
        print(f" - Prediction file: {pred_path.name}")
        print(f" - GT file:         {gt_filename}")
        
        if gt_path.exists():
            try:
                # (A) 예측값 로드
                pred = np.load(pred_path)
                # 예측 마스크가 (1, 352, 352) 형태라면 squeeze
                if pred.ndim == 3 and pred.shape[0] == 1:
                    pred = pred[0]  # -> (352, 352)
                
                # (B) 예측 마스크 이진화
                pred_binary = (pred > 0.5).astype(np.uint8)
                
                # (C) GT 로드
                gt = np.array(Image.open(gt_path))
                # 만약 3채널(RGB)라면 첫 채널만 가져옴 (흑백화)
                if gt.ndim == 3:
                    gt = gt[..., 0]
                
                # GT도 이진 마스크화 (0 or 255)
                gt_binary = (gt > 127).astype(np.uint8)

                # (D) GT 크기 확인 후 352x352가 아니면 리사이즈
                if gt_binary.shape != (352, 352):
                    gt_binary = np.array(
                        Image.fromarray(gt_binary).resize((352, 352), Image.NEAREST)
                    )

                # (E) 비교 시각화 및 저장
                save_comparison_image(pred_binary, gt_binary, base_stem, save_root)
                
                # (F) 성능 지표 계산
                dice = calculate_dice(pred_binary, gt_binary)
                iou = calculate_iou(pred_binary, gt_binary)
                polyp_size = np.sum(gt_binary)
                boundary_clarity = calculate_boundary_clarity(gt_binary)
                
                print(f" - Dice: {dice:.3f}, IoU: {iou:.3f}, Polyp Size: {polyp_size}")
                
                results.append({
                    'case_id': base_stem,
                    'dice': dice,
                    'iou': iou,
                    'polyp_size': polyp_size,
                    'boundary_clarity': boundary_clarity
                })
                
            except Exception as e:
                print(f"Error processing {pred_path.name}: {str(e)}")
        else:
            print(f"GT file not found: {gt_filename}")
    
    if not results:
        print("No matching files found!")
        return None
    
    # (G) 결과를 DataFrame으로 정리
    df = pd.DataFrame(results)
    
    # (H) 성능 분석 및 저장
    if len(df) > 0:
        analyze_performance_groups(df, save_root)
    
    return df

def save_comparison_image(pred, gt, case_id, save_root):
    """
    예측 마스크(pred)와 GT 마스크(gt)를 비교하여 시각화 / 저장
    """
    plt.figure(figsize=(12, 4))
    
    # 1) 예측 마스크
    plt.subplot(1, 3, 1)
    plt.imshow(pred, cmap='gray')
    plt.title(f'Prediction - {case_id}')
    plt.axis('off')
    
    # 2) GT 마스크
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title(f'GT - {case_id}')
    plt.axis('off')
    
    # 3) Overlay 시각화 (빨강=예측, 초록=GT, 노랑=겹침)
    plt.subplot(1, 3, 3)
    overlay = np.zeros((*pred.shape, 3), dtype=float)
    overlay[pred == 1] = [1, 0, 0]  # 빨강
    overlay[gt == 1]   = [0, 1, 0]  # 초록
    overlay[(pred == 1) & (gt == 1)] = [1, 1, 0]  # 노랑
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    # 저장
    plt.savefig(f'{save_root}/{case_id}_comparison.png')
    plt.close()

def analyze_performance_groups(df, save_root):
    """
    Dice 기준 성능 구간별로 그룹화해 통계, 시각화, 엑셀 저장.
    """
    os.makedirs(save_root, exist_ok=True)
    
    # 구간 정의
    df['performance_group'] = pd.cut(
        df['dice'], 
        bins=[-np.inf, 0.0, 0.4, 0.6, np.inf],
        labels=['완전실패', '낮은성능', '중간성능', '높은성능']
    )
    
    print("\n=== Polyp Segmentation 성능 구간 분석 ===\n")
    group_stats = df.groupby('performance_group').agg({
        'case_id': 'count',
        'dice': ['mean', 'std'],
        'iou': ['mean', 'std'],
        'polyp_size': ['mean', 'std'],
        'boundary_clarity': ['mean', 'std']
    }).round(3)
    print("구간별 통계:")
    print(group_stats)
    
    # 히스토그램 / 산점도 시각화
    plt.figure(figsize=(15, 10))
    
    # 1) Dice 분포
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='dice', bins=30, kde=True)
    plt.axvline(x=0.0, color='r', linestyle='--', alpha=0.5, label='실패')
    plt.axvline(x=0.4, color='y', linestyle='--', alpha=0.5, label='낮은성능')
    plt.axvline(x=0.6, color='g', linestyle='--', alpha=0.5, label='중간성능')
    plt.title('Dice Score Distribution')
    plt.legend()
    
    # 2) IoU 분포
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='iou', bins=30, kde=True)
    plt.title('IoU Score Distribution')
    
    # 3) Dice vs IoU
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='dice', y='iou', hue='performance_group')
    plt.title('Dice vs IoU')
    
    # 4) Polyp Size vs Dice
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='polyp_size', y='dice', hue='performance_group')
    plt.title('Polyp Size vs Dice')
    
    plt.tight_layout()
    plt.savefig(f'{save_root}/detailed_analysis.png')
    plt.close()
    
    # 엑셀 저장
    excel_path = f'{save_root}/performance_analysis.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        # 전체 결과
        df.sort_values('dice', ascending=False).to_excel(writer, sheet_name='All Cases', index=False)
        
        # 각 그룹별 시트
        for group in ['완전실패', '낮은성능', '중간성능', '높은성능']:
            group_cases = df[df['performance_group'] == group].sort_values('dice')
            group_cases.to_excel(writer, sheet_name=group, index=False)
        
        # 통계 시트
        group_stats.to_excel(writer, sheet_name='Statistics')
    
    print(f"\n분석 결과가 {excel_path}에 저장되었습니다.")
    
    # Dice Score 구간 분포 출력
    print("\n=== Dice Score 분포표 ===")
    dice_bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
    dice_labels = ['0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8 이상']
    dice_dist = pd.cut(df['dice'], bins=dice_bins, labels=dice_labels)
    dice_counts = dice_dist.value_counts().sort_index()
    print("\nDice Score 분포:")
    for label, count in dice_counts.items():
        print(f"{label}: {count}개 ({count/len(df)*100:.1f}%)")

def calculate_dice(pred, gt):
    """Dice coefficient 계산"""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)

def calculate_iou(pred, gt):
    """IoU 계산"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-6)

def calculate_boundary_clarity(mask):
    """
    경계선 선명도 계산:
    sobel 필터 등으로 경계를 추출한 뒤, 그 평균 값을 계산.
    """
    from scipy import ndimage
    gradient = ndimage.sobel(mask)
    return np.mean(gradient)

if __name__ == "__main__":
    # 예시 경로 (사용자 환경에 따라 수정)
    pred_dir = '/userHome/userhome2/donghee/modelcombination/output_EMCAD_kvasir/output_250219_031122/EMCAD_Iter_10/test_outputs'
    gt_dir   = '/userHome/userhome2/donghee/modelcombination/sessile-main-Kvasir-SEG/masks'
    output_dir = './analysis_results'
    
    df = analyze_test_results(pred_dir, gt_dir, output_dir)
