import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from torchvision import transforms
import torch.nn.functional as F
import sys
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add model path to system path
sys.path.append('/userHome/userhome2/donghee/modelcombination/models')

# Model definitions
MODEL_MODULES = {
    'CaraNet': {
        'backbone': lambda m: {
            'layer0': m.resnet.conv1,
            'layer1': m.resnet.layer1,
            'layer2': m.resnet.layer2,
            'layer3': m.resnet.layer3,
            'layer4': m.resnet.layer4
        },
        'rfb_module': lambda m: {'rfb2': m.rfb2_1, 'rfb3': m.rfb3_1, 'rfb4': m.rfb4_1},
        'cfp_module': lambda m: {'cfp1': m.CFP_1, 'cfp2': m.CFP_2, 'cfp3': m.CFP_3},
        'attention_module': lambda m: {'aa1': m.aa_kernel_1, 'aa2': m.aa_kernel_2, 'aa3': m.aa_kernel_3},
        'decoder': lambda m: m.agg1
    },
    'ConvSegNet': {
        'encoder_layers': lambda m: {
            'layer0': m.encoder.layer0,
            'layer1': m.encoder.layer1,
            'layer2': m.encoder.layer2,
            'layer3': m.encoder.layer3
        },
        'context_refine': lambda m: {
            'c1': m.c1,
            'c2': m.c2,
            'c3': m.c3,
            'c4': m.c4
        },
        'output': lambda m: m.output
    },
    'CPS': {
        'encoder': lambda m: {
            'down1': m.down1,
            'down2': m.down2,
            'down3': m.down3,
            'down4': m.down4,
            'down5': m.down5
        },
        'decoder': lambda m: {
            'up1': m.up1,
            'up2': m.up2,
            'up3': m.up3,
            'up4': m.up4,
            'up5': m.up5
        },
        'attention': lambda m: {
            'LABlock_1': m.LABlock_1,
            'LABlock_2': m.LABlock_2,
            'LABlock_3': m.LABlock_3
        },
        'fusion': lambda m: {
            'fuse1': m.fuse1,
            'fuse2': m.fuse2,
            'fuse3': m.fuse3,
            'fuse4': m.fuse4,
            'fuse5': m.fuse5
        }
    },
    'CFHA': {
        'encoder': lambda m: {
            'enc1': m.encoder1,
            'enc2': m.encoder2,
            'enc3': m.encoder3,
            'enc4': m.encoder4,
            'enc5': m.encoder5
        },
        'tdb_module': lambda m: {
            'down1': m.down1,
            'down2': m.down2,
            'down3': m.down3,
            'down4': m.down4
        },
        'sff_module': lambda m: {
            'sff1': m.sff1,
            'sff2': m.sff2,
            'sff3': m.sff3,
            'sff4': m.sff4
        },
        'attention': lambda m: {
            'ma1': m.ma1,
            'ma2': m.ma2,
            'ma3': m.ma3,
            'ma4': m.ma4
        },
        'drf': lambda m: m.drf
    },
    'Polyper': {
        'backbone': lambda m: {
            'stem': m.stages[0],
            'stage1': m.stages[1],
            'stage2': m.stages[2],
            'stage3': m.stages[3],
            'stage4': m.stages[4]
        },
        'head': lambda m: {
            'decoder': m.decoder_level,
            'squeeze': m.squeeze,
            'bottleneck': m.sep_bottleneck
        },
        'output': lambda m: {
            'align': m.align,
            'final': m.final_conv
        }
    },
'CRC': {
    'encoder': lambda m: {
        'quad_receptive': m.encoder1[0],  # ModifiedQuadReceptiveFieldModule
        'maxpool': m.encoder1[1],         # MaxPool2d
        'encoder2': m.encoder2,           # backbone.layer1
        'encoder3': m.encoder3,           # backbone.layer2
        'encoder4': m.encoder4,           # backbone.layer3
        'encoder5': m.encoder5            # backbone.layer4
    },
    'feature_processing': lambda m: {
        'glcm_block0': m.glcm_blocks[0],
        'glcm_block1': m.glcm_blocks[1],
        'glcm_block2': m.glcm_blocks[2],
        'glcm_block3': m.glcm_blocks[3],
        'glcm_block4': m.glcm_blocks[4],
        'mmca_block0': m.mmca_blocks[0],
        'mmca_block1': m.mmca_blocks[1],
        'mmca_block2': m.mmca_blocks[2],
        'mmca_block3': m.mmca_blocks[3],
        'mmca_block4': m.mmca_blocks[4],
        'asm_block0': m.asm_blocks[0],
        'asm_block1': m.asm_blocks[1],
        'asm_block2': m.asm_blocks[2],
        'asm_block3': m.asm_blocks[3],
        'asm_block4': m.asm_blocks[4]
    },
    'decoder': lambda m: {
        'decoder0': m.decoders[0],
        'decoder1': m.decoders[1],
        'decoder2': m.decoders[2],
        'decoder3': m.decoders[3],
        'pred_conv0': m.pred_convs[0],
        'pred_conv1': m.pred_convs[1],
        'pred_conv2': m.pred_convs[2],
        'pred_conv3': m.pred_convs[3],
        'pred_conv4': m.pred_convs[4],
        'final_conv': m.final_conv
    }
}
}

def verify_model(model, weight_path, device):
    """모델 구조와 가중치 파일의 호환성을 검증하는 함수"""
    try:
        if not os.path.exists(weight_path):
            print(f"Weight file not found at: {weight_path}")
            return False
            
        state_dict = torch.load(weight_path, map_location=device)
        
        model_keys = set(model.state_dict().keys())
        weight_keys = set(state_dict.keys())
        
        missing_keys = model_keys - weight_keys
        unexpected_keys = weight_keys - model_keys
        
        if missing_keys:
            print(f"Missing keys in weight file: {missing_keys}")
            return False
            
        if unexpected_keys:
            print(f"Unexpected keys in weight file: {unexpected_keys}")
            return False
            
        for key in model_keys:
            if model.state_dict()[key].shape != state_dict[key].shape:
                print(f"Shape mismatch for layer {key}:")
                print(f"Expected: {model.state_dict()[key].shape}")
                print(f"Found: {state_dict[key].shape}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")
        return False

def preprocess_image(image_path, target_size=352):
    """이미지 전처리 함수"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.ToTensor()
    image_tensor = transform(image).float()
    
    # 종횡비를 유지하면서 target_size에 맞게 리사이징
    ratio = min(target_size / image_tensor.shape[1], target_size / image_tensor.shape[2])
    new_h = int(image_tensor.shape[1] * ratio)
    new_w = int(image_tensor.shape[2] * ratio)
    
    # 패딩 대신 리사이즈로 변경
    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0), 
        size=(target_size, target_size),  # 직접 target_size로 리사이즈
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return image_tensor.unsqueeze(0), image

class FeatureMapVisualizer:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
    def hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook
        
    def register_hooks(self, module_dict):
        """등록된 모듈들에 대해 forward hook을 설정"""
        for module_type, get_modules in module_dict.items():
            modules = get_modules(self.model)
            
            if isinstance(modules, dict):
                for name, module in modules.items():
                    full_name = f"{module_type}_{name}"
                    hook = module.register_forward_hook(self.hook_fn(full_name))
                    self.hooks.append(hook)
            else:
                hook = modules.register_forward_hook(self.hook_fn(module_type))
                self.hooks.append(hook)
                
    def remove_hooks(self):
        """등록된 모든 hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _get_layer_type(self, module_name):
        """레이어 타입 판별"""
        if 'pool' in module_name.lower():
            return 'pooling'
        elif 'denseblock' in module_name.lower():
            return 'dense'
        elif 'transition' in module_name.lower():
            return 'transition'
        elif 'norm' in module_name.lower():
            return 'normalization'
        elif any(x in module_name.lower() for x in ['attention', 'aa_kernel', 'lablock', 'ma']):
            return 'attention'
        else:
            return 'default'

    def visualize_feature_maps(self, feature_maps, module_name, save_path, original_image, ground_truth):
        if not isinstance(feature_maps, torch.Tensor):
            return

        print(f"\nModule: {module_name}")
        print(f"Feature map shape: {feature_maps.shape}")
        
        # Get original image size
        orig_h, orig_w = original_image.shape[:2]
        
        # Calculate appropriate figure size
        fig_width = 16
        fig_height = fig_width / 4
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground Truth
        plt.subplot(1, 4, 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # CAM-style visualization
        feature_maps = feature_maps[0].cpu().numpy()  # 첫 번째 배치
        
        # 각 채널의 전역 평균 풀링 값 계산
        channel_weights = np.mean(feature_maps, axis=(1, 2))
        
        # CAM 생성을 위한 가중치 정규화
        channel_weights = (channel_weights - np.min(channel_weights)) / (np.max(channel_weights) - np.min(channel_weights) + 1e-8)
        
        # 가중치를 적용한 피처맵 생성
        cam = np.zeros(feature_maps[0].shape)
        for i, weight in enumerate(channel_weights):
            cam += weight * feature_maps[i]
        
        # CAM 정규화 및 후처리
        cam = np.maximum(cam, 0)  # ReLU로 음수 제거
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        # 원본 이미지 크기로 리사이즈
        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        
        # CAM visualization
        plt.subplot(1, 4, 3)
        plt.imshow(cam_resized, cmap='jet')  # 'jet' colormap for better visibility
        plt.title(f'Activation Map\n({feature_maps.shape[0]} channels)')
        plt.axis('off')
        
        # Overlay visualization
        plt.subplot(1, 4, 4)
        plt.imshow(original_image)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title('Activation Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 상위 20% 활성화 영역 시각화
        threshold = np.percentile(cam_resized, 80)
        highlighted_cam = np.copy(cam_resized)
        highlighted_cam[highlighted_cam < threshold] = 0
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.imshow(highlighted_cam, cmap='jet', alpha=0.7)
        plt.title('High Activation Regions')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(highlighted_cam, cmap='jet')
        plt.title('Activation Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        save_dir = os.path.dirname(save_path)
        plt.savefig(os.path.join(save_dir, f'{module_name}_highlighted.png'), bbox_inches='tight', dpi=300)
        plt.close()



    def _visualize_attention_heads(self, feature_maps, module_name, save_path, original_image):
        """Visualize individual attention heads"""
        plt.figure(figsize=(20, 4))
        num_heads = min(8, feature_maps.shape[0])
        
        orig_h, orig_w = original_image.shape[:2]
        
        for i in range(num_heads):
            plt.subplot(1, num_heads, i + 1)
            head_map = feature_maps[i]
            
            # Normalize head map
            head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
            
            # Resize to match original image size
            head_map_resized = cv2.resize(head_map, (orig_w, orig_h))
            
            plt.imshow(original_image)
            plt.imshow(head_map_resized, cmap='jet', alpha=0.6)
            plt.title(f'Head {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        save_dir = os.path.dirname(save_path)
        plt.savefig(os.path.join(save_dir, f'{module_name}_heads.png'))
        plt.close()

def analyze_model(model, model_name, weight_path, image_path, mask_path, output_dir):
    """모델 분석 및 시각화 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not verify_model(model, weight_path, device):
        print("모델 구조와 가중치가 일치하지 않습니다.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"Loaded weights from: {weight_path}")
    
    # 입력 이미지와 ground truth 마스크 로드
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    visualizer = FeatureMapVisualizer(model)
    model.eval()
    
    try:
        visualizer.register_hooks(MODEL_MODULES[model_name])
        
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)
            final_mask = (output > 0.5).float().squeeze().cpu().numpy()
        
        # Process each feature map
        for name, feature in visualizer.features.items():
            print(f"\nVisualizing module: {name}")
            print(f"Feature shape: {feature.shape}")
            
            save_dir = os.path.join(output_dir, model_name, os.path.splitext(os.path.basename(image_path))[0])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{name}_feature_maps.png')
            
            visualizer.visualize_feature_maps(
                feature,
                name,
                save_path,
                original_image,
                ground_truth  # ground_truth 인자 추가
            )
        
        # Save final prediction visualization
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground Truth
        plt.subplot(1, 4, 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Prediction Mask
        plt.subplot(1, 4, 3)
        # 원본 이미지 크기에 맞게 예측 마스크 리사이즈
        final_mask_resized = cv2.resize(final_mask, (original_image.shape[1], original_image.shape[0]))
        plt.imshow(final_mask_resized, cmap='gray')
        plt.title('Prediction Mask')
        plt.axis('off')
        
        # Ground Truth vs Prediction Overlay
        plt.subplot(1, 4, 4)
        # 초록색: Ground Truth, 빨간색: 예측 마스크
        overlay = np.zeros((*original_image.shape[:2], 3))
        overlay[ground_truth > 0] = [0, 1, 0]  # Ground Truth: 초록색
        overlay[final_mask_resized > 0] = [1, 0, 0]  # Prediction: 빨간색
        # 겹치는 부분은 노란색으로 표시
        overlap = (ground_truth > 0) & (final_mask_resized > 0)
        overlay[overlap] = [1, 1, 0]  # 겹치는 부분: 노란색
        plt.imshow(overlay)
        plt.title('GT vs Pred Overlay\nGreen: GT, Red: Pred, Yellow: Overlap')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_prediction.png'), bbox_inches='tight', dpi=300)
        plt.close()
            
    finally:
        visualizer.remove_hooks()

def main():
    try:
        # Import models
        from models.CaraNet import CaraNet
        from models.convsegnet import convsegnet
        from models.cps import cps
        from models.CFHA_Net import CFHA_Net
        from models.polyper import polyper
        from models.crc_real import crc_real
    except ImportError as e:
        print(f"모델 import 오류: {e}")
        print("models 디렉토리가 올바른 위치에 있는지 확인해주세요.")
        return

    # Models dictionary
    models = {
        'CaraNet': lambda: CaraNet(in_channels=3, num_classes=1),
        'ConvSegNet': lambda: convsegnet(in_channels=3, number_of_classes=1),
        'CPS': lambda: cps(in_channels=3, out_channels=1),
        'CFHA': lambda: CFHA_Net(in_channel=3, out_channel=1),
        'Polyper': lambda: polyper(in_chans=3, num_classes=1),
        'CRC': lambda: crc_real(in_channels=3, num_classes=1)
    }

    print("\n사용 가능한 모델:")
    for idx, model_name in enumerate(models.keys(), 1):
        print(f"{idx}. {model_name}")
    
    while True:
        try:
            model_idx = int(input("\n분석할 모델 번호를 선택하세요 (종료는 0): "))
            if model_idx == 0:
                break
            if model_idx < 1 or model_idx > len(models):
                print("잘못된 모델 번호입니다.")
                continue
                
            model_name = list(models.keys())[model_idx-1]
            model = models[model_name]()
            
            print("\n필요한 경로들을 입력해주세요:")
            image_path = input("분석할 이미지 경로: ")
            mask_path = input("Ground Truth 마스크 경로: ")
            weight_path = input("가중치 파일 경로: ")
            output_dir = f"feature_map_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"\n분석 시작:")
            print(f"모델: {model_name}")
            print(f"가중치 파일: {weight_path}")
            print(f"이미지: {image_path}")
            print(f"마스크: {mask_path}")
            
            analyze_model(model, model_name, weight_path, image_path, mask_path, output_dir)
            print(f"\n분석 완료. 결과가 {output_dir} 디렉토리에 저장되었습니다.")
            
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except Exception as e:
            print(f"오류 발생: {str(e)}")

if __name__ == '__main__':
    main()