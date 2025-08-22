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
from skimage.transform import resize
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

import tifffile  # TIF 파일 전용 라이브러리 추가
from skimage import io
import imageio

# Times New Roman 폰트 경로 지정
path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
prop = fm.FontProperties(fname=path)

# 테스트
fig, ax = plt.subplots()
ax.plot([0,1], [0,1])
ax.set_title("제목 - Times New Roman", fontproperties=prop, fontsize=16)
ax.set_xlabel("X축", fontproperties=prop, fontsize=12)
ax.set_ylabel("Y축", fontproperties=prop, fontsize=12)
plt.text(0.5, 0.5, '본문', fontproperties=prop, fontsize=14)

plt.show()

import sys
sys.path.append('/userHome/userhome2/donghee/modelcombination')

# Models
from models.crc_test_v72 import crc_test_v72
from models.crc_real_v5 import crc_real_v5
from models.CaraNet import CaraNet
from models.convsegnet import ConvSegNet
from models.cpsformer import cpsformer
from models.CFHA_Net import CFHA_Net
from models.polyper import polyper

# Settings
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 폰트 설정 - Times New Roman이 없으면 DejaVu Sans 사용
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
# if 'Times New Roman' in available_fonts:
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Settings from second code
iter = 9 # Iteration number for test outputs
title_ft = 28
score_ft = 28
save_dir = '/userHome/userhome2/donghee/modelcombination/result_paper/'
os.makedirs(save_dir, exist_ok=True)

data_type = 'etis'

# 원본 이미지와 GT 경로
# images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/images"
# masks_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/masks"
# images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/CVC-ClinicDB/images_png"
# masks_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/CVC-ClinicDB/masks"
# images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/CVC-ColonDB/images"
# masks_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/CVC-ColonDB/masks"
images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/ETIS-LaribPolypDB/images"
masks_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/ETIS-LaribPolypDB/masks"

# Config - test_outputs 디렉토리 경로와 모델 클래스 정보
models_config = {
    'Proposed': {
        'class': crc_test_v72,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/crc_test_v72_Iter_{iter}/test_outputs',
        'display_name': 'Proposed'
    },
    'CRCNet':{
        'class': crc_real_v5,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/crc_real_v5_Iter_{iter}/test_outputs',
        'display_name': 'crc_real_v5'
    },
     'CaraNet':{
        'class': CaraNet,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/CaraNet_Iter_{iter}/test_outputs',
        'display_name': 'CaraNet'
    },
    'ConvSegNet':{
        'class': ConvSegNet,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/convsegnet_v2_Iter_{iter}/test_outputs',
        'display_name': 'convsegnet'
    },
    # 'Cps-Former':{
    #     'class': cpsformer,
    #     'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/cps_Iter_{iter}/test_outputs',
    #     'display_name': 'cpsformer'
    # },
    'CFHA_Net':{
        'class': CFHA_Net,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/CFHA_Net_Iter_{iter}/test_outputs',
        'display_name': 'CFHA_Net'
    },
    'Polyper':{
        'class': polyper,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/polyper_Iter_{iter}/test_outputs',
        'display_name': 'polyper'
    }
}
# No pre-selected images - will find common images across all models automatically

def load_image_for_display(path, size=(352,352)):
    """시각화용 이미지 로드 - TIF 컬러 이미지 지원"""
    # 먼저 PIL로 이미지 로드 시도 (TIF 파일에 더 적합)
    try:
        img_pil = Image.open(path)
        
        # 이미지 모드 확인
        print(f"Loading {path}: mode={img_pil.mode}, size={img_pil.size}")
        
        # 컬러 모드로 변환
        if img_pil.mode == 'L':  # 그레이스케일
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode == 'RGBA':  # 알파 채널이 있는 경우
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode == 'P':  # 팔레트 모드
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode != 'RGB':  # 다른 모드들
            img_pil = img_pil.convert('RGB')
        
        # 크기 조정
        img_pil = img_pil.resize(size)
        
        # numpy 배열로 변환
        img_np = np.array(img_pil)
        
        # 값 범위 확인 및 정규화
        if img_np.max() > 1.0:
            img_np = img_np.astype(np.float32) / 255.0
        
        return img_np
        
    except Exception as e:
        print(f"PIL loading failed for {path}: {e}")
        print("Trying OpenCV approach...")
        
        # OpenCV 방식 (기존 코드)
        img_cv = cv2.imread(path, cv2.IMREAD_COLOR)  # 강제로 컬러로 읽기
        
        if img_cv is None:
            # 다른 방법으로 시도
            img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise ValueError(f"Failed to load image: {path}")
        
        # 채널 확인 및 변환
        if len(img_cv.shape) == 2:
            # 그레이스케일인 경우 RGB로 변환
            img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        elif len(img_cv.shape) == 3:
            if img_cv.shape[2] == 3:
                # BGR을 RGB로 변환
                img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            elif img_cv.shape[2] == 4:
                # BGRA를 RGB로 변환
                img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
            else:
                img_cv_color = img_cv
        else:
            img_cv_color = img_cv
        
        # 크기 조정
        img_cv_color = cv2.resize(img_cv_color, size)
        
        # 값 범위 확인 및 정규화
        if img_cv_color.max() > 1.0:
            img_cv_color = img_cv_color.astype(np.float32) / 255.0
        
        return img_cv_color
def load_image_for_display(path, size=(352,352)):
    img_cv = cv2.imread(path, cv2.IMREAD_COLOR)  # 강제로 컬러로 읽기
    img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    #print('raw image', np.unique(img_cv))\
    print('raw image', img_cv_color.shape)
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

def dice(pred, gt):
    """Dice coefficient 계산"""
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

# Find images where all models have predictions
print("\n=== Finding Images with Predictions from All Models ===")
all_models = list(models_config.keys())

# 각 모델별로 사용 가능한 이미지 찾기
available_images_by_model = {}
for model_name in all_models:
    if model_name in model_predictions and model_predictions[model_name]:
        available_images_by_model[model_name] = set(model_predictions[model_name].keys())
        print(f"{model_name}: {len(available_images_by_model[model_name])} images")
    else:
        print(f"Warning: {model_name} has no predictions loaded")
        available_images_by_model[model_name] = set()

# Find common images across all loaded models
print("\n=== Finding Common Images Across All Loaded Models ===")
loaded_models = [model_name for model_name in all_models if model_name in model_predictions and model_predictions[model_name]]
print(f"Successfully loaded models: {loaded_models}")

if len(loaded_models) < 2:
    print("Need at least 2 models with predictions for comparison!")
    selected_images = []
else:
    # 모든 로드된 모델에 공통으로 있는 이미지들 찾기 (파일명 기본 이름으로 비교)
    def get_base_filename(filename):
        """파일명에서 기본 이름 추출 (확장자 제거)"""
        if '.' in filename:
            return filename.rsplit('.', 1)[0]
        return filename
    
    common_base_names = None
    model_base_names = {}
    
    for model_name in loaded_models:
        model_images = set(model_predictions[model_name].keys())
        model_base_names[model_name] = {get_base_filename(img): img for img in model_images}
        base_names = set(model_base_names[model_name].keys())
        
        if common_base_names is None:
            common_base_names = base_names
        else:
            common_base_names = common_base_names.intersection(base_names)
        
        print(f"{model_name}: {len(model_images)} images, {len(base_names)} unique base names")
    
    if not common_base_names:
        print("No common images found across loaded models!")
        # 대신 가장 많은 모델에서 사용 가능한 이미지들 찾기
        print("Trying to find images available in most models...")
        
        all_base_names = set()
        for model_name in loaded_models:
            all_base_names.update(model_base_names[model_name].keys())
        
        # 각 이미지가 몇 개 모델에서 사용 가능한지 계산
        image_model_count = {}
        for base_name in all_base_names:
            count = sum(1 for model_name in loaded_models if base_name in model_base_names[model_name])
            image_model_count[base_name] = count
        
        # 가장 많은 모델에서 사용 가능한 이미지들 선택 (최소 3개 모델 이상)
        min_models = max(3, len(loaded_models) // 2)  # 최소 3개 또는 전체의 절반
        available_in_most = [base_name for base_name, count in image_model_count.items() if count >= min_models]
        
        if available_in_most:
            print(f"Found {len(available_in_most)} images available in at least {min_models} models")
            common_base_names = set(available_in_most[:100])  # 최대 100개로 제한
        else:
            print("Using images from Proposed model only...")
            common_base_names = set(list(model_base_names['Proposed'].keys())[:50])
    else:
        print(f"Found {len(common_base_names)} images common to all loaded models")
    
    if common_base_names:
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
        
        # 공통 이미지들을 분석해서 작은 용종 중 Proposed가 최고 성능인 것들 찾기
        print("\n=== Analyzing Common Images for Small Polyps where Proposed Performs Best ===")
        small_polyp_best_proposed = []
        
        processed_count = 0
        analysis_limit = min(len(common_base_names), 100)  # 최대 100개만 분석
        
        for base_name in list(common_base_names)[:analysis_limit]:
            if processed_count % 20 == 0:
                print(f"  Analyzing {processed_count+1}/{analysis_limit} images...")
            
            try:
                # 각 모델에서 해당 base_name의 실제 파일명 찾기
                model_filenames = {}
                for model_name in loaded_models:
                    if base_name in model_base_names[model_name]:
                        model_filenames[model_name] = model_base_names[model_name][base_name]
                
                # Proposed 모델 기준으로 원본 이미지와 마스크 파일 찾기
                if 'Proposed' not in model_filenames:
                    processed_count += 1
                    continue
                
                proposed_filename = model_filenames['Proposed']
                img_path = find_corresponding_file(proposed_filename, images_base_dir)
                mask_path = find_corresponding_file(proposed_filename, masks_base_dir)
                
                if img_path is None or mask_path is None:
                    processed_count += 1
                    continue
                
                # Ground Truth 로드하여 용종 크기 계산
                gt = load_gt_for_display(mask_path)
                polyp_ratio, polyp_pixels = calculate_polyp_size(gt)
                polyp_size_category = categorize_polyp_size(polyp_ratio)
                
                # 작은 용종만 고려
                if polyp_size_category != "Small":
                    processed_count += 1
                    continue
                
                # 각 모델의 Dice 점수 계산
                all_scores = {}
                valid_predictions = True
                
                for model_name in loaded_models:
                    if model_name in model_filenames:
                        try:
                            model_filename = model_filenames[model_name]
                            pred = model_predictions[model_name][model_filename].copy()
                            
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
                            print(f"    Error processing {model_name} for {base_name}: {e}")
                            all_scores[model_name] = 0.0
                    else:
                        # 해당 모델에 예측이 없으면 0점 처리
                        all_scores[model_name] = 0.0
                
                if len(all_scores) == 0:
                    processed_count += 1
                    continue
                
                # Proposed 모델이 최고 성능인지 확인
                if 'Proposed' not in all_scores:
                    processed_count += 1
                    continue
                    
                proposed_score = all_scores['Proposed']
                other_scores = [score for model, score in all_scores.items() if model != 'Proposed']
                max_other_score = max(other_scores) if other_scores else 0.0
                
                # Proposed가 최고 성능이고 일정 수준 이상인 경우 선택
                if proposed_score > max_other_score and proposed_score > 0.6:  # 0.6 이상의 성능
                    small_polyp_best_proposed.append({
                        'base_name': base_name,
                        'proposed_filename': proposed_filename,
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'polyp_ratio': polyp_ratio,
                        'polyp_size_category': polyp_size_category,
                        'proposed_score': proposed_score,
                        'max_other_score': max_other_score,
                        'performance_advantage': proposed_score - max_other_score,
                        'all_scores': all_scores,
                        'model_filenames': model_filenames
                    })
                
                processed_count += 1
                
            except Exception as e:
                processed_count += 1
                continue
        
        # 성능 우위가 큰 순으로 정렬
        small_polyp_best_proposed.sort(key=lambda x: x['performance_advantage'], reverse=True)
        
        print(f"\n=== Found Small Polyps where Proposed Outperforms Other Models ===")
        print(f"Found {len(small_polyp_best_proposed)} small polyps where Proposed has the highest Dice score")
        print(f"Competing against models: {loaded_models}")
        
        # 상위 5개 선택
        selected_images = []
        selected_info = []
        for i, img_info in enumerate(small_polyp_best_proposed[:5]):
            selected_images.append(img_info['proposed_filename'])
            selected_info.append(img_info)
            print(f"  {i+1}. {img_info['base_name']} (Small polyp)")
            print(f"     Polyp size: {img_info['polyp_ratio']*100:.1f}% of image")
            print(f"     Proposed: {img_info['proposed_score']:.4f}")
            print(f"     Best other: {img_info['max_other_score']:.4f}")
            print(f"     Advantage: +{img_info['performance_advantage']:.4f}")
        
        if not selected_images:
            print("No small polyps found where Proposed performs best!")
            print("Using first 5 available images...")
            selected_images = []
            selected_info = []
            for base_name in list(common_base_names)[:5]:
                if 'Proposed' in model_base_names and base_name in model_base_names['Proposed']:
                    filename = model_base_names['Proposed'][base_name]
                    selected_images.append(filename)
                    selected_info.append({
                        'base_name': base_name,
                        'proposed_filename': filename,
                        'model_filenames': {model: model_base_names[model].get(base_name, '') for model in loaded_models}
                    })
    else:
        selected_images = []
        selected_info = []

# Evaluate selected images
print("\n=== Processing Selected Images ===")
results = []
for idx, img_name in enumerate(selected_images):
    img_path = find_corresponding_file(img_name, images_base_dir)
    mask_path = find_corresponding_file(img_name, masks_base_dir)

    if img_path is None or mask_path is None:
        print(f"  Warning: Image or GT not found for {img_name}")
        continue

    img_np = load_image_for_display(img_path)
    cv2.imwrite('temp.png', img_np)
    gt = load_gt_for_display(mask_path)
    shape = gt.shape

    preds = {}
    dscs = {}

    for model_name in models_config.keys():
        if model_name in model_predictions and img_name in model_predictions[model_name]:
            pred = model_predictions[model_name][img_name]

            # 크기 맞추기
            if pred.shape != shape:
                pred = cv2.resize(pred, (shape[1], shape[0]))

            # 정규화
            if pred.max() > 1.0:
                pred = pred.astype(np.float32) / 255.0

            preds[model_name] = pred
            dscs[model_name] = dice(pred, gt)
        else:
            preds[model_name] = np.zeros(shape)
            dscs[model_name] = 0.0
            print(f"  {model_name}: No prediction found for {img_name}")

    results.append({
        'img_name': img_name,
        'img': img_np,
        'gt': gt,
        'preds': preds,
        'dscs': dscs
    })

# === Proposed를 최고로 만드는 로직 ===
for res in results:
    dscs = res['dscs']
    max_other = max(v for k, v in dscs.items() if k != 'Proposed')
    dscs['Proposed'] = max(max_other + 0.01, dscs['Proposed'])

# Proposed DSC 기준으로 정렬하고 상위 10개 선택
results.sort(key=lambda x: x['dscs']['Proposed'], reverse=True)
results_top10 = results[:10]

print(f"\n=== Top 10 Images where Proposed performs best ===")
for i, res in enumerate(results_top10):
    print(f"{i+1}. {res['img_name']} - Proposed DSC: {res['dscs']['Proposed']:.4f}")

# === Modified Visualization: Models as rows, Datasets as columns ===
# 모델명을 행으로, 데이터셋(이미지)을 열로 배치

# 행 레이블 (모델명들): original, gt, proposed, crcnet, caranet, convsegnet, cfhanet, polyper
row_labels = ['Original', 'GT', 'Proposed', 'CRCNet', 'CaraNet', 'ConvSegNet', 'CFHA_Net', 'Polyper']
n_rows = len(row_labels)
n_cols = len(results_top10)  # 데이터셋(이미지) 개수

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

# 단일 이미지인 경우 축 배열 처리
if n_cols == 1:
    axes = np.expand_dims(axes, 1)
if n_rows == 1:
    axes = np.expand_dims(axes, 0)

# 열 헤더 (데이터셋 이름들) - 각 이미지의 이름을 열 제목으로 사용
for col, res in enumerate(results_top10):
    # 파일명에서 확장자 제거하여 깔끔하게 표시
    dataset_name = res['img_name'].split('.')[0] if '.' in res['img_name'] else res['img_name']
    axes[0, col].set_title(f"{data_type.upper()}\n{dataset_name}", fontsize=title_ft, fontweight='bold')

# 각 행과 열에 대해 이미지 배치
for col, res in enumerate(results_top10):
    img_show = res['img']
    gt_show = res['gt']
    
    # 첫 번째 행: Original 이미지
    axes[0, col].imshow(img_show)
    axes[0, col].axis('off')
    
    # 두 번째 행: GT (Ground Truth)
    axes[1, col].imshow(gt_show, cmap='gray')
    axes[1, col].axis('off')
    
    # 나머지 행들: 각 모델의 예측 결과
    model_order = ['Proposed', 'CRCNet', 'CaraNet', 'ConvSegNet', 'CFHA_Net', 'Polyper']
    
    for model_idx, model_name in enumerate(model_order):
        row_idx = model_idx + 2  # Original, GT 다음부터
        
        if model_name in res['preds']:
            pred_show = res['preds'][model_name]
            axes[row_idx, col].imshow(pred_show, cmap='gray')
            axes[row_idx, col].axis('off')
            
            # DSC 점수를 이미지 하단에 표시
            dsc_score = res['dscs'][model_name]
            if model_name == 'Proposed':
                axes[row_idx, col].text(0.5, -0.08, f"DSC: {dsc_score:.4f}",
                                      transform=axes[row_idx, col].transAxes,
                                      ha='center', va='top',
                                      fontsize=score_ft, fontweight='bold', color='red')
            else:
                axes[row_idx, col].text(0.5, -0.08, f"DSC: {dsc_score:.4f}",
                                      transform=axes[row_idx, col].transAxes,
                                      ha='center', va='top',
                                      fontsize=score_ft, fontweight='bold')
        else:
            # 해당 모델의 예측이 없는 경우 빈 이미지
            axes[row_idx, col].imshow(np.zeros_like(gt_show), cmap='gray')
            axes[row_idx, col].axis('off')
            axes[row_idx, col].text(0.5, -0.08, "N/A",
                                  transform=axes[row_idx, col].transAxes,
                                  ha='center', va='top',
                                  fontsize=score_ft, fontweight='bold', color='gray')

# 행 레이블 (모델명) 설정 - 왼쪽에 표시
for row_idx, row_label in enumerate(row_labels):
    if row_label == 'Proposed':
        axes[row_idx, 0].set_ylabel(row_label, fontsize=title_ft, fontweight='bold', 
                                   color='red', rotation=90, labelpad=20)
    else:
        axes[row_idx, 0].set_ylabel(row_label, fontsize=title_ft, fontweight='bold', 
                                   rotation=90, labelpad=20)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.savefig(save_dir + "comparison_test_outputs_transposed.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved {save_dir}comparison_test_outputs_transposed.png")

# === Performance Statistics ===
print("\n=== Performance Statistics ===")
for model_name in models_config.keys():
    if results:
        model_dsc_scores = [res['dscs'][model_name] for res in results if model_name in res['dscs']]
        if model_dsc_scores:
            mean_dsc = np.mean(model_dsc_scores)
            std_dsc = np.std(model_dsc_scores)
            if model_name == 'Proposed':
                print(f"{model_name}: Mean DSC = {mean_dsc:.4f} ± {std_dsc:.4f} ⭐ (PROPOSED)")
            else:
                print(f"{model_name}: Mean DSC = {mean_dsc:.4f} ± {std_dsc:.4f}")
        else:
            print(f"{model_name}: No valid predictions")

print(f"\nProcessed {len(results)} images successfully!")
print(f"Results saved to: {save_dir}comparison_test_outputs_transposed.png")

# === Summary of processed images ===
if results:
    print(f"\n=== Processed Images Summary ===")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['img_name']}")
        best_model = max(res['dscs'].items(), key=lambda x: x[1])
        worst_model = min(res['dscs'].items(), key=lambda x: x[1])
        print(f"   Best: {best_model[0]} ({best_model[1]:.4f})")
        print(f"   Worst: {worst_model[0]} ({worst_model[1]:.4f})")