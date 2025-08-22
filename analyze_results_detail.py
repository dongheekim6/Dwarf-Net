import os
import numpy as np
from PIL import Image
import glob
from collections import defaultdict
from scipy import stats
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# 기존의 helper 함수들은 그대로 유지
def calculate_metrics(pred, gt):
    """모든 평가 지표 계산"""
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    
    return {
        'dice': dice,
        'iou': iou,
        'f1_score': f1_score, 
        'precision': precision,
        'recall': recall
    }

def get_mask_ratio(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_pixels = np.sum(mask > 0)
    return (mask_pixels / total_pixels) * 100

def resize_and_preprocess(img, target_size=(352, 352)):
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize(target_size, Image.NEAREST)
    return np.array(img) / 255.0

def get_performance_category(metric_value):
    if metric_value <= 0.2:
        return 'complete_failure'
    elif metric_value <= 0.4:
        return 'low_performance'
    elif metric_value <= 0.6:
        return 'medium_performance'
    else:
        return 'high_performance'

def create_comparison_visualization(gt_img, pred_img, save_path):
    """Ground Truth와 예측 결과 비교 시각화"""
    gt_viz = (gt_img * 255).astype(np.uint8)
    pred_viz = (pred_img * 255).astype(np.uint8)
    
    overlay = np.zeros((gt_img.shape[0], gt_img.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = gt_viz  # Ground Truth를 초록색으로
    overlay[..., 2] = pred_viz  # Prediction을 파란색으로
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_viz, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(pred_viz, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay\nGreen: GT, Blue: Pred')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_model_outputs(gt_dir, model_pred_dirs, model_name):
    """단일 모델의 여러 반복 실험 결과 분석"""
    results = {
        'under_50': defaultdict(dict),
        'under_25': defaultdict(dict),
        'under_10': defaultdict(dict),
        'under_5': defaultdict(dict)
    }
    
    performance_results = {
        'complete_failure': [],
        'low_performance': [],
        'medium_performance': [],
        'high_performance': []
    }
    
    processed_images = set()
    gt_paths = []
    gt_paths.extend(glob.glob(os.path.join(gt_dir, '*.jpg')))
    gt_paths.extend(glob.glob(os.path.join(gt_dir, '*.png')))
    
    for pred_dir in model_pred_dirs:
        for gt_path in gt_paths:
            base_name = os.path.splitext(os.path.basename(gt_path))[0]
            pred_path = os.path.join(pred_dir, f'{base_name}.png.npy')
            
            if not os.path.exists(pred_path) or base_name in processed_images:
                continue
            
            try:
                gt_img = np.array(Image.open(gt_path).convert('L'))
                gt_img = resize_and_preprocess(gt_img)
                gt_img = (gt_img > 0).astype(np.float32)
                
                pred = np.load(pred_path)
                if pred.shape[0] == 1:
                    pred = pred[0]
                pred = resize_and_preprocess(pred)
                pred = (pred > 0.5).astype(np.float32)
                
                print(f"Processing {model_name} - {base_name}")
                
                mask_ratio = get_mask_ratio(gt_img)
                metrics = calculate_metrics(pred, gt_img)
                
                performance_category = get_performance_category(metrics['dice'])
                performance_results[performance_category].append({
                    'image_name': base_name,
                    'dice_score': metrics['dice'],
                    'iou_score': metrics['iou'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'mask_ratio': mask_ratio,
                    'gt_img': gt_img,
                    'pred_img': pred
                })
                
                if mask_ratio <= 50:
                    results['under_50'][base_name] = metrics
                if mask_ratio <= 25:
                    results['under_25'][base_name] = metrics
                if mask_ratio <= 10:
                    results['under_10'][base_name] = metrics
                if mask_ratio <= 5:
                    results['under_5'][base_name] = metrics
                    
                processed_images.add(base_name)
                
            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
                continue
                
    return results, performance_results

def save_to_excel(results, performance_results, model_name, output_dir):
    """결과를 Excel 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'{model_name}_analysis_{timestamp}.xlsx')
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # 성능 구간별 통계
        performance_stats = []
        for category in ['complete_failure', 'low_performance', 'medium_performance', 'high_performance']:
            category_results = performance_results[category]
            if category_results:
                stats_dict = {
                    '성능 구간': category,
                    '이미지 수': len(category_results),
                    '평균 Dice': np.mean([r['dice_score'] for r in category_results]),
                    'Dice 표준오차': stats.sem([r['dice_score'] for r in category_results]) if len(category_results) > 1 else 0,
                    '평균 IoU': np.mean([r['iou_score'] for r in category_results]),
                    'IoU 표준오차': stats.sem([r['iou_score'] for r in category_results]) if len(category_results) > 1 else 0,
                }
                performance_stats.append(stats_dict)
        
        # 개별 이미지 성능
        performance_data = []
        for category in performance_results:
            for result in performance_results[category]:
                performance_data.append({
                    '성능 구간': category,
                    '이미지 이름': result['image_name'],
                    'Dice': result['dice_score'],
                    'IoU': result['iou_score'],
                    'F1': result['f1_score'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    '마스크 비율(%)': result['mask_ratio']
                })
        
        # 데이터프레임 생성 및 저장
        stats_df = pd.DataFrame(performance_stats)
        details_df = pd.DataFrame(performance_data)
        
        stats_df.to_excel(writer, sheet_name='성능구간통계', index=False)
        details_df.to_excel(writer, sheet_name='상세결과', index=False)
        
        # 엑셀 서식 설정
        workbook = writer.book
        num_format = workbook.add_format({'num_format': '0.0000'})
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:H', 15, num_format)
    
    print(f"\nExcel 파일이 저장되었습니다: {output_file}")
    return output_file

def save_visualizations(performance_results, model_name, output_dir):
    """시각화 결과 저장"""
    viz_dir = os.path.join(output_dir, f'visualizations_{model_name}')
    
    for category in performance_results.keys():
        category_dir = os.path.join(viz_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        for result in performance_results[category]:
            save_path = os.path.join(category_dir, f"{result['image_name']}_comparison.png")
            create_comparison_visualization(result['gt_img'], result['pred_img'], save_path)
    
    print(f"\n시각화 결과가 저장되었습니다: {viz_dir}")
    return viz_dir

if __name__ == "__main__":
    # Ground Truth 디렉토리 설정
    gt_dir = '/userHome/userhome2/donghee/modelcombination/CVC-ColonDB/masks'
    
    # 하나의 모델에 대한 10개의 test_outputs 경로 설정
    model_pred_dirs = [
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_1/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_2/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_3/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_4/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_5/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_6/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_7/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_8/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_9/test_outputs',
        '/userHome/userhome2/donghee/modelcombination/output_haesung_test/output_250520_004026_haesung_v25/haesung_v25_Iter_10/test_outputs'
    ]
    
    model_name = 'haesung_v25'  # 분석할 모델의 이름
    output_dir = 'results_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 모델 분석 실행
    print(f"\n분석 중: {model_name}")
    results, performance_results = analyze_model_outputs(gt_dir, model_pred_dirs, model_name)

    # 결과를 Excel 파일로 저장
    excel_file = save_to_excel(results, performance_results, model_name, output_dir)

    # 시각화 결과 저장
    viz_dir = save_visualizations(performance_results, model_name, output_dir)

    # 콘솔에 간단한 요약 출력
    print(f"\n{model_name} 성능 구간 분포:")
    for category in ['complete_failure', 'low_performance', 'medium_performance', 'high_performance']:
        print(f"  {category}: {len(performance_results[category])}개")