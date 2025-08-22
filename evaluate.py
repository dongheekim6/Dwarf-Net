import os
import numpy as np
from PIL import Image
import glob
from collections import defaultdict
from scipy import stats
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def calculate_metrics(pred, gt):
   """모든 평가 지표 계산"""
   # True Positives, False Positives, False Negatives 계산
   tp = np.sum(pred * gt)  # intersection
   fp = np.sum(pred * (1 - gt))  # pred에서 1이지만 gt에서 0인 경우
   fn = np.sum((1 - pred) * gt)  # pred에서 0이지만 gt에서 1인 경우
   
   # Precision과 Recall 계산
   precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
   recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
   
   # F1 Score 계산 (precision과 recall의 조화평균)
   f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
   
   # Dice 계산 
   dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0
   
   # IoU 계산
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
   """이미지 크기 조정 및 전처리"""
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
               
               # 성능 구간별 분류 (Dice 기준)
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

def analyze_mask_ratio_performance(performance_results):
    """마스크 비율 구간별 성능 분석 (누적 방식)"""
    ratio_ranges = {
        '0-50%': (0, 50),
        '0-25%': (0, 25),
        '0-10%': (0, 10),
        '0-5%': (0, 5)
    }
    
    ratio_stats = {}
    for range_name, (min_ratio, max_ratio) in ratio_ranges.items():
        metrics_sum = defaultdict(list)
        
        for category in performance_results.values():
            for result in category:
                mask_ratio = result['mask_ratio']
                if min_ratio <= mask_ratio < max_ratio:
                    metrics_sum['dice'].append(result['dice_score'])
                    metrics_sum['iou'].append(result['iou_score'])
                    metrics_sum['f1'].append(result['f1_score'])
                    metrics_sum['precision'].append(result['precision'])
                    metrics_sum['recall'].append(result['recall'])
        
        if metrics_sum['dice']:  # 결과가 있는 경우만
            ratio_stats[range_name] = {
                '이미지 수': len(metrics_sum['dice']),
                '평균 Dice': np.mean(metrics_sum['dice']),
                'Dice 표준오차': stats.sem(metrics_sum['dice']) if len(metrics_sum['dice']) > 1 else 0,
                '평균 IoU': np.mean(metrics_sum['iou']),
                'IoU 표준오차': stats.sem(metrics_sum['iou']) if len(metrics_sum['iou']) > 1 else 0,
                '평균 F1': np.mean(metrics_sum['f1']),
                'F1 표준오차': stats.sem(metrics_sum['f1']) if len(metrics_sum['f1']) > 1 else 0,
                '평균 Precision': np.mean(metrics_sum['precision']),
                'Precision 표준오차': stats.sem(metrics_sum['precision']) if len(metrics_sum['precision']) > 1 else 0,
                '평균 Recall': np.mean(metrics_sum['recall']),
                'Recall 표준오차': stats.sem(metrics_sum['recall']) if len(metrics_sum['recall']) > 1 else 0
            }
    
    return ratio_stats

def find_common_failure_cases(all_models_results, threshold=0.4):
   """모든 모델에서 공통적으로 성능이 좋지 않은 케이스 찾기"""
   common_failures = defaultdict(list)
   
   all_images = set()
   model_scores = {}
   
   for model_name, (_, performance_results) in all_models_results.items():
       model_scores[model_name] = {}
       for category in performance_results.values():
           for result in category:
               img_name = result['image_name']
               all_images.add(img_name)
               model_scores[model_name][img_name] = {
                   'dice': result['dice_score'],
                   'iou': result['iou_score'],
                   'f1': result['f1_score'],
                   'precision': result['precision'],
                   'recall': result['recall']
               }
   
   for img_name in all_images:
       scores = []
       models_with_score = []
       for model_name in model_scores:
           if img_name in model_scores[model_name]:
               scores.append(model_scores[model_name][img_name]['dice'])
               models_with_score.append(model_name)
       
       if scores and max(scores) < threshold:
           for model_name, score in zip(models_with_score, scores):
               common_failures[img_name].append({
                   'model': model_name,
                   'scores': model_scores[model_name][img_name]
               })
   
   return common_failures

def save_to_excel_multi_model(all_models_results, common_failures, output_dir):
   """결과를 Excel 파일로 저장"""
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_file = os.path.join(output_dir, f'multi_model_analysis_{timestamp}.xlsx')
   
   with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
       for model_name, (_, performance_results) in all_models_results.items():
           # 성능 구간별 통계
           performance_stats = []
           for category in ['complete_failure', 'low_performance', 'medium_performance', 'high_performance']:
               category_results = performance_results[category]
               if category_results:
                   performance_stats.append({
                        '성능 구간': category,
                        '이미지 수': len(category_results),
                        '평균 Dice': np.mean([r['dice_score'] for r in category_results]),
                        'Dice 표준오차': stats.sem([r['dice_score'] for r in category_results]) if len(category_results) > 1 else 0,
                        '평균 IoU': np.mean([r['iou_score'] for r in category_results]),
                        'IoU 표준오차': stats.sem([r['iou_score'] for r in category_results]) if len(category_results) > 1 else 0,
                        '평균 F1': np.mean([r['f1_score'] for r in category_results]),
                        'F1 표준오차': stats.sem([r['f1_score'] for r in category_results]) if len(category_results) > 1 else 0,
                        '평균 Precision': np.mean([r['precision'] for r in category_results]),
                        'Precision 표준오차': stats.sem([r['precision'] for r in category_results]) if len(category_results) > 1 else 0,
                        '평균 Recall': np.mean([r['recall'] for r in category_results]),
                        'Recall 표준오차': stats.sem([r['recall'] for r in category_results]) if len(category_results) > 1 else 0
                   })
           
           # 마스크 비율별 통계
           ratio_stats = analyze_mask_ratio_performance(performance_results)
           ratio_stats_list = []
           for range_name, stats_dict in ratio_stats.items():
               stats_dict['마스크 비율 구간'] = range_name
               ratio_stats_list.append(stats_dict)
           
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
           ratio_stats_df = pd.DataFrame(ratio_stats_list)
           details_df = pd.DataFrame(performance_data)
           
           stats_df.to_excel(writer, sheet_name=f'{model_name}_성능구간통계', index=False)
           ratio_stats_df.to_excel(writer, sheet_name=f'{model_name}_비율별통계', index=False)
           details_df.to_excel(writer, sheet_name=f'{model_name}_상세', index=False)
       
       # 공통 실패 케이스 시트 생성
       common_failures_data = []
       for img_name, results in common_failures.items():
           for result in results:
               common_failures_data.append({
                   '이미지 이름': img_name,
                   '모델': result['model'],
                   'Dice': result['scores']['dice'],
                   'IoU': result['scores']['iou'],
                   'F1': result['scores']['f1'],
                   'Precision': result['scores']['precision'],
                   'Recall': result['scores']['recall']
               })
       
       common_failures_df = pd.DataFrame(common_failures_data)
       if not common_failures_df.empty:
           common_failures_df = common_failures_df.sort_values(['이미지 이름', '모델'])
           common_failures_df.to_excel(writer, sheet_name='공통_실패_케이스', index=False)
       
       workbook = writer.book
       num_format = workbook.add_format({'num_format': '0.0000'})
       
       for sheet_name in writer.sheets:
           worksheet = writer.sheets[sheet_name]
           worksheet.set_column('A:A', 20)
           worksheet.set_column('B:H', 15, num_format)
   
   print(f"\nExcel 파일이 저장되었습니다: {output_file}")
   return output_file
def save_visualizations_multi_model(all_models_results, common_failures, output_dir):
    """시각화 결과 저장"""
    viz_dir = os.path.join(output_dir, 'visualizations_etis_test')
    
    # 각 모델별 시각화
    for model_name, (_, performance_results) in all_models_results.items():
        model_viz_dir = os.path.join(viz_dir, model_name)
        for category in performance_results.keys():
            os.makedirs(os.path.join(model_viz_dir, category), exist_ok=True)
            for result in performance_results[category]:
                save_path = os.path.join(model_viz_dir, category, 
                                       f"{result['image_name']}_comparison.png")
                create_comparison_visualization(result['gt_img'], result['pred_img'], save_path)
    
    # 공통 실패 케이스 시각화
    if common_failures:
        common_failures_dir = os.path.join(viz_dir, 'common_failures')
        os.makedirs(common_failures_dir, exist_ok=True)
        
        for img_name in common_failures.keys():
            for model_name, (_, performance_results) in all_models_results.items():
                for category in performance_results.values():
                    for result in category:
                        if result['image_name'] == img_name:
                            save_path = os.path.join(common_failures_dir, 
                                                   f"{img_name}_{model_name}_comparison.png")
                            create_comparison_visualization(result['gt_img'], 
                                                         result['pred_img'], 
                                                         save_path)
    
    print(f"\n시각화 결과가 저장되었습니다: {viz_dir}")
    return viz_dir

# 사용 예시
if __name__ == "__main__":

# 사용 예시
    gt_dir = '/userHome/userhome2/donghee/modelcombination/ETIS-LaribPolypDB/masks'

    # 각 모델별 test_outputs 경로 설정
    model_dirs = {
        'Model1': [
            '/userHome/userhome2/donghee/modelcombination/output_CaraNet_etis/output_250221_062235/CaraNet_Iter_1/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_CaraNet_etis/output_250221_062235/CaraNet_Iter_2/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_CaraNet_etis/output_250221_062235/CaraNet_Iter_3/test_outputs'
        ],
        'Model2': [
            '/userHome/userhome2/donghee/modelcombination/output_convsegnet_etis/output_250221_042141/convsegnet_Iter_1/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_convsegnet_etis/output_250221_042141/convsegnet_Iter_2/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_convsegnet_etis/output_250221_042141/convsegnet_Iter_3/test_outputs'
        ],
        'Model3': [
            '/userHome/userhome2/donghee/modelcombination/output_cps_etis/output_250221_095937/cps_Iter_1/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_cps_etis/output_250221_095937/cps_Iter_2/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_cps_etis/output_250221_095937/cps_Iter_3/test_outputs'
        ],
        'Model4': [
            '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_etis/output_250221_061941/CFHA_Net_Iter_1/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_etis/output_250221_061941/CFHA_Net_Iter_2/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_etis/output_250221_061941/CFHA_Net_Iter_3/test_outputs'
        ],
        'Model5': [
            '/userHome/userhome2/donghee/modelcombination/output_polyper_etis/output_250224_012752/polyper_Iter_1/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_polyper_etis/output_250224_012752/polyper_Iter_2/test_outputs',
            '/userHome/userhome2/donghee/modelcombination/output_polyper_etis/output_250224_012752/polyper_Iter_3/test_outputs'
        ],
            'Model6': [
                '/userHome/userhome2/donghee/modelcombination/output_crc_test_제안기본/output_250416_155213/crc_test_v4_Iter_1/test_outputs',
                '/userHome/userhome2/donghee/modelcombination/output_crc_test_제안기본/output_250416_155213/crc_test_v4_Iter_2/test_outputs',
                '/userHome/userhome2/donghee/modelcombination/output_crc_test_제안기본/output_250416_155213/crc_test_v4_Iter_3/test_outputs'
            ]
    }
    output_dir = 'results_comparisons'
    os.makedirs(output_dir, exist_ok=True)

    # 각 모델별 분석 실행
    all_models_results = {}
    for model_name, pred_dirs in model_dirs.items():
        if pred_dirs:  # 경로가 설정된 모델만 분석
            print(f"\n분석 중: {model_name}")
            results, performance_results = analyze_model_outputs(gt_dir, pred_dirs, model_name)
            all_models_results[model_name] = (results, performance_results)

    # 공통 실패 케이스 찾기
    common_failures = find_common_failure_cases(all_models_results)

    # 결과를 Excel 파일로 저장
    excel_file = save_to_excel_multi_model(all_models_results, common_failures, output_dir)

    # 시각화 결과 저장
    viz_dir = save_visualizations_multi_model(all_models_results, common_failures, output_dir)

    # 콘솔에 간단한 요약 출력
    print("\n각 모델별 성능 구간 분포:")
    for model_name, (_, performance_results) in all_models_results.items():
        print(f"\n{model_name}:")
        for category in ['complete_failure', 'low_performance', 'medium_performance', 'high_performance']:
            print(f"  {category}: {len(performance_results[category])}개")

    print(f"\n공통 실패 케이스 수: {len(common_failures)}")
    if common_failures:
        print("\n공통 실패 케이스:")
        for img_name in common_failures.keys():
            print(f"  {img_name}")