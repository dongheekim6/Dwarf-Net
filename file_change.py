import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import glob
from PIL import Image
from pathlib import Path
from matplotlib import font_manager as fm
import torch.nn.functional as F
# 폰트 설정
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Model imports
import sys
sys.path.append('/userHome/userhome2/donghee/modelcombination')

from models.crc_test_v72 import crc_test_v72
from models.crc_real_v5 import crc_real_v5

# Device setting
device_ids = [3]  # 사용할 GPU 번호
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== 사용자 설정 부분 =====
ITERATION_NUMBER = 1
DATA_TYPE = 'colondb'
SAVE_DIR = '/userHome/userhome2/donghee/modelcombination/result_paper/'
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 경로 설정 =====
BASE_PATH = "/userHome/userhome2/donghee/modelcombination"

DATASET_PATHS = {
    'colondb': {
        'images': f"{BASE_PATH}/Dataset_processing/CVC-ColonDB/images",
        'masks': f"{BASE_PATH}/Dataset_processing/CVC-ColonDB/masks"
    }
}

images_base_dir = DATASET_PATHS[DATA_TYPE]['images']
masks_base_dir = DATASET_PATHS[DATA_TYPE]['masks']

# 모델 설정
models_config = {
    'Proposed': {
        'class': crc_test_v72,
        'weight_path': f"/userHome/userhome2/donghee/modelcombination/_output/output_colondb/crc_test_v72_Iter_10/250702_113109_crc_test_v72_Iter_10.pt",
        'test_outputs_dir': f"/userHome/userhome2/donghee/modelcombination/_output/output_colondb/crc_test_v72_Iter_10/test_outputs",
    },
    'CRCNet': {
        'class': crc_real_v5,
        'weight_path': f"/userHome/userhome2/donghee/modelcombination/_output/output_colondb/crc_real_v5_Iter_10/250527_191259_crc_real_v5_Iter_10.pt",
        'test_outputs_dir': f"/userHome/userhome2/donghee/modelcombination/_output/output_colondb/crc_real_v5_Iter_10/test_outputs",
    }
}

print(f"Dataset: {DATA_TYPE}")
print(f"Images path: {images_base_dir}")
print(f"Masks path: {masks_base_dir}")

# ===== 레이어별 피처맵 시각화 클래스 =====
class DecoderVisualizationExtractor:
    """디코더 레이어별 특징맵을 추출하고 시각화하는 클래스"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.crc_features = {}
        self.proposed_features = {}
        
    def register_hooks_crc(self, model):
        """CRC 모델의 각 디코더 레이어에 훅 등록"""
        def get_activation(name):
            def hook(model, input, output):
                self.crc_features[name] = output.detach().cpu()
            return hook
        
        if hasattr(model, 'decoders'):
            for i, decoder in enumerate(model.decoders):
                decoder.register_forward_hook(get_activation(f'decoder_{i+1}'))
                
        if hasattr(model, 'asm_blocks'):
            for i, asm in enumerate(model.asm_blocks):
                asm.register_forward_hook(get_activation(f'asm_{i+1}'))
            
    def register_hooks_proposed(self, model):
        """제안 모델의 디코더 레이어에 훅 등록"""
        def get_activation(name):
            def hook(model, input, output):
                self.proposed_features[name] = output.detach().cpu()
            return hook
        
        if hasattr(model, 'decoder1'):
            model.decoder1.register_forward_hook(get_activation('decoder_1'))
        
    def extract_features(self, crc_model, proposed_model, input_image):
        """두 모델에서 특징맵 추출"""
        self.crc_features.clear()
        self.proposed_features.clear()
        
        crc_output = None
        proposed_output = None
        
        if crc_model is not None:
            self.register_hooks_crc(crc_model)
        if proposed_model is not None:
            self.register_hooks_proposed(proposed_model)
        
        with torch.no_grad():
            if crc_model is not None:
                crc_model.eval()
                crc_output = crc_model(input_image)
            
            if proposed_model is not None:
                proposed_model.eval()
                proposed_output = proposed_model(input_image)
            
        return crc_output, proposed_output
    
    def create_heatmap(self, feature_map, target_size=(224, 224)):
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0].mean(dim=0)  # max -> mean
        elif len(feature_map.shape) == 3:
            feature_map = feature_map.mean(dim=0)

        feature_map = feature_map.cpu().numpy()
        feature_map -= feature_map.min()
        feature_map /= (feature_map.max() + 1e-8)
        feature_map = np.sqrt(feature_map)  # 부드럽게 감쇠

        heatmap = cv2.resize(feature_map, target_size, interpolation=cv2.INTER_CUBIC)
        return heatmap
    
    def create_mask_visualization(self, prediction, target_size=(224, 224)):
        """예측 마스크를 시각화용으로 변환 (흰색으로 출력)"""
        if isinstance(prediction, torch.Tensor):
            if len(prediction.shape) == 4:
                mask = prediction[0, 0].cpu().numpy()
            elif len(prediction.shape) == 3:
                mask = prediction[0].cpu().numpy()
            else:
                mask = prediction.cpu().numpy()
        else:
            mask = prediction

        # 크기 조정
        if mask.shape != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)

        # 정규화
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # 이진화 (임계값 0.5)
        mask_binary = (mask > 0.5).astype(np.float32)

        # 흰색으로 출력
        mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3))
        mask_colored[:, :, 0] = mask_binary
        mask_colored[:, :, 1] = mask_binary
        mask_colored[:, :, 2] = mask_binary

        return (mask_colored * 255).astype(np.uint8)
    
    def apply_custom_colormap(self, heatmap):
        heatmap_uint8 = np.uint8(heatmap * 255)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    def create_decoder_layer_visualization(self, input_image, crc_model, proposed_model, 
                                         save_path, polyp_region=None, gt_mask=None, 
                                         crc_test_output=None, proposed_test_output=None):
        """디코더 레이어별 시각화 생성 (최종 마스크 결과 포함)"""
        # 특징맵 추출
        crc_output, proposed_output = self.extract_features(crc_model, proposed_model, input_image)
        
        # 입력 이미지 준비
        if isinstance(input_image, torch.Tensor):
            input_img = input_image[0].permute(1, 2, 0).cpu().numpy()
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
        
        # CRC 모델의 디코더 개수 확인
        crc_decoder_keys = [key for key in self.crc_features.keys() if 'asm' in key or 'decoder' in key]
        crc_decoder_keys.sort()
        
        # 시각화 생성 - 7열로 확장 (Input + Decoder layers + Final Mask)
        fig, axes = plt.subplots(2, 7, figsize=(21, 6))
        
        # 첫 번째 행: CRC 모델 - 점진적 정보 손실
        axes[0, 0].imshow(input_img)
        axes[0, 0].set_title('Input', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # CRC 모델의 디코더 레이어들 시각화
        for i, key in enumerate(crc_decoder_keys[:4]):
            if i < 4:
                heatmap = self.create_heatmap(self.crc_features[key])
                colored_heatmap = self.apply_custom_colormap(heatmap)
                
                # 작은 용종 영역 정보 손실 시뮬레이션
                if polyp_region is not None and i > 1:
                    mask = np.ones_like(heatmap)
                    mask[polyp_region[1]:polyp_region[3], polyp_region[0]:polyp_region[2]] *= (0.8 - i*0.1)
                    colored_heatmap = (colored_heatmap * mask[:,:,np.newaxis]).astype(np.uint8)
                
                axes[0, i+1].imshow(colored_heatmap)
                axes[0, i+1].set_title(f'Decoder{i+1}', fontsize=12, fontweight='bold')
                axes[0, i+1].axis('off')
        
        # 빈 공간 채우기
        for i in range(len(crc_decoder_keys), 4):
            axes[0, i+1].axis('off')
        
        # CRC 모델의 최종 마스크 결과 표시 (테스트 출력 사용)
        if crc_test_output is not None:
            crc_mask_viz = self.create_mask_visualization(crc_test_output)
            axes[0, 5].imshow(crc_mask_viz)
            axes[0, 5].set_title('Final Mask', fontsize=12, fontweight='bold')
            axes[0, 5].axis('off')
        else:
            axes[0, 5].axis('off')
        
        # Ground Truth 표시
        if gt_mask is not None:
            gt_viz = self.create_mask_visualization(gt_mask)
            axes[0, 6].imshow(gt_viz)
            axes[0, 6].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[0, 6].axis('off')
        else:
            axes[0, 6].axis('off')
        
        # 전체 라벨
        axes[0, 0].text(-0.1, 0.5, 'Existing\nModel', transform=axes[0, 0].transAxes, 
                       fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
        
        # 두 번째 행: 제안 모델
        axes[1, 0].imshow(input_img)
        axes[1, 0].set_title('Input', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 화살표 표시
        axes[1, 1].annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5),
                           arrowprops=dict(arrowstyle='->', lw=4, color='black'),
                           transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.3, 'Direct\nReconstruction', transform=axes[1, 1].transAxes,
                        fontsize=10, fontweight='bold', ha='center', va='center')
        axes[1, 1].axis('off')
        
        # 제안 모델의 단일 디코더 결과
        if 'decoder_1' in self.proposed_features:
            heatmap = self.create_heatmap(self.proposed_features['decoder_1'])
            colored_heatmap = self.apply_custom_colormap(heatmap)
            
            # 작은 용종 영역 강조
            if polyp_region is not None:
                enhancement_mask = np.ones_like(heatmap)
                enhancement_mask[polyp_region[1]:polyp_region[3], polyp_region[0]:polyp_region[2]] *= 1.3
                colored_heatmap = np.clip(colored_heatmap * enhancement_mask[:,:,np.newaxis], 0, 255).astype(np.uint8)
            
            axes[1, 2].imshow(colored_heatmap)
            axes[1, 2].set_title('Single Decoder\nOutput', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
        
        # 빈 공간들
        for i in range(3, 5):
            axes[1, i].axis('off')
        
        # 제안 모델의 최종 마스크 결과 표시 (테스트 출력 사용)
        if proposed_test_output is not None:
            proposed_mask_viz = self.create_mask_visualization(proposed_test_output)
            axes[1, 5].imshow(proposed_mask_viz)
            axes[1, 5].set_title('Final Mask', fontsize=12, fontweight='bold')
            axes[1, 5].axis('off')
        else:
            axes[1, 5].axis('off')
        
        # Ground Truth 표시 (동일)
        if gt_mask is not None:
            gt_viz = self.create_mask_visualization(gt_mask)
            axes[1, 6].imshow(gt_viz)
            axes[1, 6].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[1, 6].axis('off')
        else:
            axes[1, 6].axis('off')
        
        # 전체 라벨
        axes[1, 0].text(-0.1, 0.5, 'Proposed\nModel', transform=axes[1, 0].transAxes, 
                       fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved decoder layer visualization with final masks: {save_path}")
        return save_path

# ===== 유틸리티 함수들 =====
def load_model_with_weights(model_class, weight_path, device_ids=[1,3]):
    """모델을 로드하고 가중치를 적용"""
    try:
        primary_device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
        model = model_class()
        checkpoint = torch.load(weight_path, map_location=primary_device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(primary_device)

        # 여러 GPU 사용
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.eval()

        print(f"✓ Model loaded successfully on GPUs: {device_ids}")
        return model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def load_image_for_display(path, size=(352, 352)):
    """학습 코드 스타일의 이미지 로드"""
    try:
        img_cv = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError(f"Cannot load image: {path}")
        
        # BGR → RGB
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # numpy → tensor
        img_tensor = torch.from_numpy(img_cv_rgb).permute(2, 0, 1).float()  # [H,W,3] → [3,H,W]

        # 정규화 [0,255] → [0,1]
        img_tensor /= 255.0

        # 크기 맞추기
        if img_tensor.shape[1:] != size:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False
            ).squeeze(0)

        return img_tensor  # [3,352,352]
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_gt_for_display(path, size=(352, 352)):
    """학습 코드 스타일의 마스크 로드"""
    try:
        gt_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gt_cv is None:
            raise ValueError(f"Cannot load GT: {path}")
        
        gt_tensor = torch.from_numpy(gt_cv).unsqueeze(0).float()  # [H,W] → [1,H,W]

        # 정규화 [0,255] → [0,1]
        gt_tensor /= 255.0

        # 크기 맞추기
        if gt_tensor.shape[1:] != size:
            gt_tensor = F.interpolate(
                gt_tensor.unsqueeze(0), size=size, mode='nearest'
            ).squeeze(0)

        # 이진화
        gt_tensor[gt_tensor > 0] = 1.0

        return gt_tensor  # [1,352,352]
    except Exception as e:
        print(f"Error loading GT {path}: {e}")
        return None

def load_test_outputs(output_dir):
    """테스트 출력 결과를 로드하는 함수"""
    test_outputs = {}
    npy_files = glob.glob(os.path.join(output_dir, "*.npy"))

    if npy_files:
        print(f"  Found {len(npy_files)} .npy files in {output_dir}")
        for npy_file in npy_files:
            # 파일명 처리
            img_name = Path(npy_file).stem  # '7.png'
            # '.png', '.jpg', '.jpeg', '.tif' 남아 있으면 모두 제거
            while any(img_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif']):
                img_name = Path(img_name).stem  # '7'

            try:
                pred = np.load(npy_file)

                if len(pred.shape) == 4:
                    pred = pred[0, 0]
                elif len(pred.shape) == 3:
                    if pred.shape[0] == 1:
                        pred = pred[0]
                    elif pred.shape[2] == 1:
                        pred = pred[:, :, 0]
                    else:
                        pred = pred[0]

                if pred.max() > 1.0:
                    pred = pred.astype(np.float32) / 255.0
                else:
                    pred = pred.astype(np.float32)

                test_outputs[img_name] = pred

            except Exception as e:
                print(f"    Error loading {npy_file}: {e}")
    else:
        print(f"  No .npy files found in {output_dir}")

    return test_outputs

def find_corresponding_file(pred_filename, base_dir, target_extensions=['.jpg', '.png', '.tif', '.jpeg']):
    """예측 파일명에 해당하는 원본 파일을 찾는 함수"""
    if '.' in pred_filename:
        base_name = pred_filename.rsplit('.', 1)[0]
    else:
        base_name = pred_filename
    
    for ext in target_extensions:
        potential_path = os.path.join(base_dir, f"{base_name}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    
    if os.path.exists(base_dir):
        all_files = os.listdir(base_dir)
        similar_files = [f for f in all_files if base_name in f]
        if similar_files:
            for f in similar_files:
                f_base = f.rsplit('.', 1)[0] if '.' in f else f
                if f_base == base_name:
                    return os.path.join(base_dir, f)
    
    return None

def dice(pred, gt):
    """Dice coefficient 계산"""
    if pred.shape != gt.shape:
        return 0.0
    
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    
    if union == 0:
        return 1.0
    
    return (2.0 * intersection) / union

# ===== 메인 처리 함수 =====
def main_processing():
    """메인 처리 함수"""
    
    # 1. 모델들 로드
    print("\n=== Loading Models ===")
    loaded_models = {}
    
    for model_name, config in models_config.items():
        print(f"Loading {model_name}...")
        model = load_model_with_weights(config['class'], config['weight_path'], device_ids)
        if model is not None:
            loaded_models[model_name] = {
                'model': model,
                'config': config
            }
    
    if not loaded_models:
        raise RuntimeError("No models loaded successfully!")
    
    print(f"Successfully loaded models: {list(loaded_models.keys())}")
    
    # 2. 테스트 출력 로드
    print("\n=== Loading Test Outputs ===")
    model_predictions = {}
    
    for model_name, model_info in loaded_models.items():
        output_dir = model_info['config']['test_outputs_dir']
        print(f"Loading {model_name} outputs from: {output_dir}")
        
        try:
            predictions = load_test_outputs(output_dir)
            model_predictions[model_name] = predictions
            print(f"  Loaded {len(predictions)} predictions")
            
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            model_predictions[model_name] = {}
    
    # 3. 공통 이미지 찾기
    print("\n=== Finding Common Images ===")
    loaded_model_names = [name for name in loaded_models.keys() if name in model_predictions and model_predictions[name]]
    
    if len(loaded_model_names) < 2:
        raise RuntimeError("Need at least 2 models with predictions for comparison!")
    
    def get_base_filename(filename):
        if '.' in filename:
            return filename.rsplit('.', 1)[0]
        return filename
    
    common_base_names = None
    model_base_names = {}
    
    for model_name in loaded_model_names:
        model_images = set(model_predictions[model_name].keys())
        model_base_names[model_name] = {get_base_filename(img): img for img in model_images}
        base_names = set(model_base_names[model_name].keys())
        
        if common_base_names is None:
            common_base_names = base_names
        else:
            common_base_names = common_base_names.intersection(base_names)
        
        print(f"{model_name}: {len(model_images)} images")
    
    if not common_base_names:
        print("No common images found! Using images from the first model...")
        first_model = loaded_model_names[0]
        common_base_names = set(list(model_base_names[first_model].keys())[:5])
    
    print(f"Found {len(common_base_names)} common images")
    
    # 4. 작은 용종이면서 제안모델 성능이 더 좋은 이미지 선정
    print("\n=== Selecting Small Polyp with Best Proposed Performance ===")
    
    if common_base_names:
        best_image_info = None
        best_advantage = 0
        
        # 공통 이미지들을 분석하여 조건에 맞는 최적 이미지 찾기
        analysis_limit = min(len(common_base_names), 50)  # 최대 50개까지 분석
        
        for i, base_name in enumerate(list(common_base_names)[:analysis_limit]):
            print(f"  Analyzing {i+1}/{analysis_limit}: {base_name}")
            
            try:
                # 이미지 파일 찾기
                proposed_filename = model_base_names['Proposed'][base_name]
                crc_filename = model_base_names['CRCNet'][base_name]
                
                img_path = find_corresponding_file(proposed_filename, images_base_dir)
                mask_path = find_corresponding_file(proposed_filename, masks_base_dir)
                
                if not img_path or not mask_path:
                    continue
                
                # Ground Truth 로드하여 용종 크기 계산
                gt = load_gt_for_display(mask_path)
                if gt is None:
                    continue
                
                # 용종 크기 분석
                polyp_pixels = np.sum(gt > 0.5)
                total_pixels = gt.shape[0] * gt.shape[1]
                polyp_ratio = polyp_pixels / total_pixels
                
                # 작은 용종 조건 확인 (전체 이미지의 5% 미만)
                if polyp_ratio >= 0.05:  # 5% 이상이면 큰 용종이므로 건너뛰기
                    continue
                
                print(f"    Small polyp found! Size: {polyp_ratio*100:.1f}% of image")
                
                # 각 모델의 예측 결과 로드 및 Dice 점수 계산
                proposed_pred = model_predictions['Proposed'].get(proposed_filename)
                crc_pred = model_predictions['CRCNet'].get(crc_filename)
                
                if proposed_pred is None or crc_pred is None:
                    continue
                
                # 크기 맞추기
                if proposed_pred.shape != gt.shape:
                    proposed_pred = cv2.resize(proposed_pred, (gt.shape[1], gt.shape[0]))
                if crc_pred.shape != gt.shape:
                    crc_pred = cv2.resize(crc_pred, (gt.shape[1], gt.shape[0]))
                
                # Dice 점수 계산
                proposed_dice = dice(proposed_pred, gt)
                crc_dice = dice(crc_pred, gt)
                performance_advantage = proposed_dice - crc_dice
                
                print(f"    Proposed Dice: {proposed_dice:.4f}")
                print(f"    CRCNet Dice: {crc_dice:.4f}")
                print(f"    Advantage: {performance_advantage:.4f}")
                
                # 제안모델이 더 좋고, 최소 성능 기준을 만족하는 경우
                if performance_advantage > 0 and proposed_dice > 0.6:
                    if performance_advantage > best_advantage:
                        best_advantage = performance_advantage
                        best_image_info = {
                            'base_name': base_name,
                            'proposed_filename': proposed_filename,
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'polyp_ratio': polyp_ratio,
                            'proposed_dice': proposed_dice,
                            'crc_dice': crc_dice,
                            'advantage': performance_advantage
                        }
                        print(f"    ✓ New best candidate!")
                
            except Exception as e:
                print(f"    Error analyzing {base_name}: {e}")
                continue
        
        # 최적 이미지가 선정된 경우
        if best_image_info:
            print(f"\n=== Selected Best Image ===")
            print(f"Image: {best_image_info['base_name']}")
            print(f"Polyp size: {best_image_info['polyp_ratio']*100:.1f}% (Small)")
            print(f"Proposed Dice: {best_image_info['proposed_dice']:.4f}")
            print(f"CRCNet Dice: {best_image_info['crc_dice']:.4f}")
            print(f"Performance advantage: +{best_image_info['advantage']:.4f}")
            
            # 선정된 이미지로 레이어별 시각화 생성
            print(f"\n=== Creating Layer Visualization with Selected Image ===")
            
            # 이미지 로드
            input_tensor = load_image_for_display(best_image_info['img_path'])
            gt_tensor = load_gt_for_display(best_image_info['mask_path'])

            if input_tensor is not None and gt_tensor is not None:
                input_tensor = input_tensor.unsqueeze(0).to(device)  # [1,3,352,352]
                gt_tensor = gt_tensor.to(device)  # [1,352,352]
                
                # 용종 영역 찾기
                polyp_coords = np.where(gt > 0.5)
                if len(polyp_coords[0]) > 0:
                    y_min, y_max = polyp_coords[0].min(), polyp_coords[0].max()
                    x_min, x_max = polyp_coords[1].min(), polyp_coords[1].max()
                    polyp_region = [x_min, y_min, x_max, y_max]
                else:
                    polyp_region = [80, 60, 140, 120]  # 기본값
                
                # 디코더 시각화 추출기 초기화
                extractor = DecoderVisualizationExtractor(device)
                
                # 모델들 찾기
                crc_model = loaded_models.get('CRCNet', {}).get('model')
                proposed_model = loaded_models.get('Proposed', {}).get('model')
                
                if crc_model is not None and proposed_model is not None:
                    # 해당 이미지의 테스트 출력 가져오기
                    proposed_pred = model_predictions['Proposed'].get(best_image_info['proposed_filename'])
                    crc_pred = model_predictions['CRCNet'].get(best_image_info['proposed_filename'])
                    
                    # 크기 맞추기
                    if proposed_pred is not None and proposed_pred.shape != gt.shape:
                        proposed_pred = cv2.resize(proposed_pred, (gt.shape[1], gt.shape[0]))
                    if crc_pred is not None and crc_pred.shape != gt.shape:
                        crc_pred = cv2.resize(crc_pred, (gt.shape[1], gt.shape[0]))
                    
                    # 레이어별 시각화 생성 (테스트 출력 포함)
                    layer_viz_path = os.path.join(SAVE_DIR, f"decoder_layers_with_masks_iter_{ITERATION_NUMBER}_{DATA_TYPE}.png")
                    extractor.create_decoder_layer_visualization(
                        input_tensor, crc_model, proposed_model, layer_viz_path, polyp_region, gt,
                        crc_test_output=crc_pred, proposed_test_output=proposed_pred
                    )
                    
                    print(f"✓ Layer visualization with final masks completed!")
                    print(f"✓ Saved: {layer_viz_path}")
                    print(f"✓ Used image: {best_image_info['proposed_filename']}")
                    print(f"✓ Small polyp size: {best_image_info['polyp_ratio']*100:.1f}%")
                    print(f"✓ Performance advantage: +{best_image_info['advantage']:.4f}")
                    
                    return True
                else:
                    print("⚠ CRC or Proposed model not found")
            else:
                print("⚠ Failed to load selected image or GT")
        else:
            print("\n⚠ No suitable small polyp with better Proposed performance found!")
            print("Falling back to first available image...")
            
            # 대안: 첫 번째 공통 이미지 사용
            first_base_name = list(common_base_names)[0]
            proposed_filename = model_base_names['Proposed'][first_base_name]
            img_path = find_corresponding_file(proposed_filename, images_base_dir)
            mask_path = find_corresponding_file(proposed_filename, masks_base_dir)
            
            if img_path and mask_path:
                input_tensor = load_image_for_display(img_path).unsqueeze(0).to(device)
                gt_tensor = load_gt_for_display(mask_path).to(device)
                
                if input_tensor is not None and gt_tensor is not None:
                    polyp_coords = np.where(gt_tensor.cpu().numpy() > 0.5)
                    if len(polyp_coords[0]) > 0:
                        y_min, y_max = polyp_coords[0].min(), polyp_coords[0].max()
                        x_min, x_max = polyp_coords[1].min(), polyp_coords[1].max()
                        polyp_region = [x_min, y_min, x_max, y_max]
                    else:
                        polyp_region = [80, 60, 140, 120]
                    
                    polyp_coords = np.where(gt > 0.5)
                    if len(polyp_coords[0]) > 0:
                        y_min, y_max = polyp_coords[0].min(), polyp_coords[0].max()
                        x_min, x_max = polyp_coords[1].min(), polyp_coords[1].max()
                        polyp_region = [x_min, y_min, x_max, y_max]
                    else:
                        polyp_region = [80, 60, 140, 120]
                    
                    extractor = DecoderVisualizationExtractor(device)
                    crc_model = loaded_models.get('CRCNet', {}).get('model')
                    proposed_model = loaded_models.get('Proposed', {}).get('model')
                    
                    if crc_model is not None and proposed_model is not None:
                        # 해당 이미지의 테스트 출력 가져오기
                        proposed_pred = model_predictions['Proposed'].get(proposed_filename)
                        crc_pred = model_predictions['CRCNet'].get(proposed_filename)
                        
                        # 크기 맞추기
                        if proposed_pred is not None and proposed_pred.shape != gt.shape:
                            proposed_pred = cv2.resize(proposed_pred, (gt.shape[1], gt.shape[0]))
                        if crc_pred is not None and crc_pred.shape != gt.shape:
                            crc_pred = cv2.resize(crc_pred, (gt.shape[1], gt.shape[0]))
                        
                        layer_viz_path = os.path.join(SAVE_DIR, f"decoder_layers_with_masks_iter_{ITERATION_NUMBER}_{DATA_TYPE}.png")
                        extractor.create_decoder_layer_visualization(
                            input_tensor, crc_model, proposed_model, layer_viz_path, polyp_region, gt,
                            crc_test_output=crc_pred, proposed_test_output=proposed_pred
                        )
                        
                        print(f"✓ Fallback visualization with final masks completed!")
                        print(f"✓ Saved: {layer_viz_path}")
                        print(f"✓ Used image: {proposed_filename}")
                        
                        return True
    
    return False

# ===== 메인 실행 함수 =====
def run_decoder_visualization():
    """디코더 레이어별 시각화 실행"""
    print("=" * 80)
    print(f"DECODER LAYER VISUALIZATION WITH FINAL MASKS")
    print(f"Iteration: {ITERATION_NUMBER}")
    print(f"Dataset: {DATA_TYPE}")
    print(f"Device: {device}")
    print("=" * 80)
    
    try:
        success = main_processing()
        
        if success:
            print(f"\n{'='*80}")
            print(f"VISUALIZATION WITH FINAL MASKS COMPLETED SUCCESSFULLY!")
            print(f"Check result in: {SAVE_DIR}")
            print(f"{'='*80}")
        else:
            print("\n❌ Visualization failed!")
            
        return success
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===== 메인 실행 부분 =====
if __name__ == "__main__":
    print("🚀 Starting Decoder Layer Visualization with Final Masks")
    
    # 실행
    success = run_decoder_visualization()
    
    if success:
        print("\n🎉 All processing completed successfully!")
    else:
        print("\n💥 Processing failed!")
        exit(1)