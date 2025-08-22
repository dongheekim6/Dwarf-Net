import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def analyze_all_cases(pred_dir, gt_dir):
    """모든 케이스의 성능을 분석하고 정렬하여 출력"""
    results = []
    pred_files = sorted(Path(pred_dir).glob('*.png.npy'))
    
    for pred_path in pred_files:
        case_num = pred_path.stem.split('.')[0]  # .png.npy 제거하고 숫자만 추출
        gt_path = Path(gt_dir) / f"{case_num}.tif"
        
        if gt_path.exists():
            # 예측값과 Ground Truth 로드
            pred = np.load(pred_path)
            gt = np.array(Image.open(gt_path))
            
            # shape 맞추기
            if gt.shape != pred.shape:
                gt = gt.transpose()
            
            # 이진화
            pred = (pred > 0.5).astype(np.uint8)
            gt = (gt > 0).astype(np.uint8)
            
            # 성능 계산
            dice = calculate_dice(pred, gt)
            iou = calculate_iou(pred, gt)
            tumor_size = np.sum(gt)
            
            # 성능 구간 결정
            if dice == 0:
                performance = "완전실패"
            elif dice < 0.117:
                performance = "낮은성능"
            elif dice < 0.394:
                performance = "중간성능"
            else:
                performance = "높은성능"
            
            results.append({
                'case_id': case_num,
                'dice': dice,
                'iou': iou,
                'tumor_size': tumor_size,
                'performance': performance
            })
    
    # DataFrame 생성 및 정렬
    df = pd.DataFrame(results)
    df = df.sort_values('dice', ascending=True)  # dice 점수로 정렬
    
    # 성능 구간별 출력
    print("\n=== 전체 케이스 성능 분석 ===")
    print(f"총 케이스 수: {len(df)}")
    print("\n성능 구간별 케이스 수:")
    print(df['performance'].value_counts())
    
    print("\n=== 케이스별 상세 성능 ===")
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    print(df[['case_id', 'dice', 'iou', 'tumor_size', 'performance']])
    
    return df

def calculate_dice(pred, gt):
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)

def calculate_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-6)

if __name__ == "__main__":
    pred_dir = '/userHome/userhome2/donghee/modelcombination/output_polyper_etis/output_250221_210333/polyper_Iter_2/test_outputs'
    gt_dir = '/userHome/userhome2/donghee/modelcombination/ETIS-LaribPolypDB/masks'
    
    df = analyze_all_cases(pred_dir, gt_dir)