# 1. Import Section
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# 2. Path Settings
WEIGHT_PATHS = {
    'CaraNet': {
        'CVC-ClinicDB': '/userHome/userhome2/donghee/modelcombination/output_CaraNet_cvc/output_250221_101504/CaraNet_Iter_1/250221_101507_CaraNet_Iter_1.pt',
        'Kvasir-SEG': '/userHome/userhome2/donghee/modelcombination/output_CaraNet_kvasir/output_250221_062401/CaraNet_Iter_1/250221_062405_CaraNet_Iter_1.pt',
        'ETIS-LaribPolypDB': '/userHome/userhome2/donghee/modelcombination/output_CaraNet_colon/output_250221_101601/CaraNet_Iter_1/250221_101605_CaraNet_Iter_1.pt',
        'CVC-ColonDB': '/userHome/userhome2/donghee/modelcombination/output_CaraNet_etis/output_250221_062235/CaraNet_Iter_2/250221_073658_CaraNet_Iter_2.pt'
    },
    'ConvSegNet': {
        'CVC-ClinicDB': '/userHome/userhome2/donghee/modelcombination/output_convsegnet_cvc/output_250221_005947/convsegnet_Iter_1/250221_005947_convsegnet_Iter_1.pt',
        'Kvasir-SEG': '/userHome/userhome2/donghee/modelcombination/output_convsegnet_kvasir/output_250221_010415/convsegnet_Iter_2/250221_022925_convsegnet_Iter_2.pt',
        'ETIS-LaribPolypDB': '/userHome/userhome2/donghee/modelcombination/output_convsegnet_colon/output_250221_004412/convsegnet_Iter_1/250221_004414_convsegnet_Iter_1.pt',
        'CVC-ColonDB': '/userHome/userhome2/donghee/modelcombination/output_convsegnet_etis/output_250221_042141/convsegnet_Iter_1/250221_042143_convsegnet_Iter_1.pt'
    },
    'CPS': {
        'CVC-ClinicDB': '/userHome/userhome2/donghee/modelcombination/output_cps_cvc/output_250221_095741/cps_Iter_2/250221_114251_cps_Iter_2.pt',
        'Kvasir-SEG': '/userHome/userhome2/donghee/modelcombination/output_cps_kvasir/output_250221_100654/cps_Iter_1/250221_100654_cps_Iter_1.pt',
        'ETIS-LaribPolypDB': '/userHome/userhome2/donghee/modelcombination/output_cps_colon/output_250221_060655/cps_Iter_1/250221_060655_cps_Iter_1.pt',
        'CVC-ColonDB': '/userHome/userhome2/donghee/modelcombination/output_cps_etis/output_250221_095937/cps_Iter_2/250221_104008_cps_Iter_2.pt'
    },
    'CFHA': {
        'CVC-ClinicDB': '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_cvc/output_250220_204920/CFHA_Net_Iter_1/250220_204921_CFHA_Net_Iter_1.pt',
        'Kvasir-SEG': '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_kvasir/output_250221_022623/CFHA_Net_Iter_1/250221_022624_CFHA_Net_Iter_1.pt',
        'ETIS-LaribPolypDB': '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_colon/output_250220_212342/CFHA_Net_Iter_1/250220_212345_CFHA_Net_Iter_1.pt',
        'CVC-ColonDB': '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_etis/output_250221_061941/CFHA_Net_Iter_2/250221_075635_CFHA_Net_Iter_2.pt'
    },
    'Polyper': {
        'CVC-ClinicDB': '/userHome/userhome2/donghee/modelcombination/output_polyper_cvc/output_250221_210450/polyper_Iter_1/250221_210450_polyper_Iter_1.pt',
        'Kvasir-SEG': '/userHome/userhome2/donghee/modelcombination/output_polyper_kvasir/output_250224_012842/polyper_Iter_1/250224_012843_polyper_Iter_1.pt',
        'ETIS-LaribPolypDB': '/userHome/userhome2/donghee/modelcombination/output_polyper_colon/output_250224_012928/polyper_Iter_1/250224_012928_polyper_Iter_1.pt',
        'CVC-ColonDB': '/userHome/userhome2/donghee/modelcombination/output_polyper_etis/output_250224_012752/polyper_Iter_2/250224_014738_polyper_Iter_2.pt'
    },
    'CRC': {
        'CVC-ClinicDB': '/userHome/userhome2/donghee/modelcombination/output_crc_cvc/output_250307_134919/crc_Iter_1/250307_134919_crc_Iter_1.pt',
        'Kvasir-SEG': '/userHome/userhome2/donghee/modelcombination/output_crc_kvasir/output_250306_231055/crc_Iter_2/250306_235234_crc_Iter_2.pt',
        'ETIS-LaribPolypDB': '/userHome/userhome2/donghee/modelcombination/output_crc_etis/output_250306_231357/crc_Iter_2/250306_234808_crc_Iter_2.pt',
        'CVC-ColonDB': '/userHome/userhome2/donghee/modelcombination/output_crc_colon/output_250306_231239/crc_Iter_3/250307_005352_crc_Iter_3.pt'
    }
}

DATASET_PATHS = {
    'CVC-ClinicDB': {
        'images': '/userHome/userhome2/donghee/modelcombination/CVC-ClinicDB/images',
        'masks': '/userHome/userhome2/donghee/modelcombination/CVC-ClinicDB/masks'
    },
    'Kvasir-SEG': {
        'images': '/userHome/userhome2/donghee/modelcombination/Kvasir-SEG/images',
        'masks': '/userHome/userhome2/donghee/modelcombination/Kvasir-SEG/masks'
    },
    'ETIS-LaribPolypDB': {
        'images': '/userHome/userhome2/donghee/modelcombination/ETIS-LaribPolypDB/images',
        'masks': '/userHome/userhome2/donghee/modelcombination/ETIS-LaribPolypDB/masks'
    },
    'CVC-ColonDB': {
        'images': '/userHome/userhome2/donghee/modelcombination/CVC-ColonDB/images',
        'masks': '/userHome/userhome2/donghee/modelcombination/CVC-ColonDB/masks'
    }
}

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
            'block0': m.features[0],      # conv0 + norm0 + relu0
            'pool0': m.features[1],       # pool0
            'denseblock1': m.features[2], # first dense block
            'transition1': m.features[3], # first transition
            'denseblock2': m.features[4], # second dense block
            'transition2': m.features[5], # second transition
            'denseblock3': m.features[6], # third dense block
            'transition3': m.features[7], # third transition
            'denseblock4': m.features[8], # final dense block
            'norm5': m.features[9]        # final norm
        },
        'decoder': lambda m: {
            'upconv1': m.decoder[0],  # 1664 -> 832
            'upconv2': m.decoder[2],  # 832 -> 416
            'upconv3': m.decoder[4],  # 416 -> 208
            'upconv4': m.decoder[6],  # 208 -> 104
            'upconv5': m.decoder[8]   # 104 -> num_classes
        }
    }

}

# 4. Utility Functions
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

def imread_kor(filePath, mode=cv2.IMREAD_UNCHANGED): 
    """한글 경로를 지원하는 이미지 읽기 함수"""
    stream = open(filePath.encode("utf-8"), "rb") 
    bytes = bytearray(stream.read()) 
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray, mode)

def add_padding(image, target_size=352):
    """이미지에 패딩을 추가하는 함수"""
    if len(image.shape) == 3:
        c, h, w = image.shape
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    else:
        h, w = image.shape
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return F.pad(image.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0).squeeze(0)

def preprocess_image(image_path, target_size=352):
    """이미지 전처리 함수"""
    image_bgr = imread_kor(image_path)
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

def calculate_mask_ratio(mask_path):
    """마스크의 전체 픽셀 대비 용종 영역 비율 계산"""
    mask = imread_kor(mask_path, cv2.IMREAD_GRAYSCALE)
    total_pixels = mask.shape[0] * mask.shape[1]
    polyp_pixels = np.sum(mask > 0)
    return (polyp_pixels / total_pixels) * 100

def get_small_polyp_dataset(dataset_name):
    """용종 크기가 5% 이하인 데이터셋만 필터링"""
    dataset_info = DATASET_PATHS[dataset_name]
    image_files = sorted(os.listdir(dataset_info['images']))
    small_polyp_data = []
    
    for img_file in image_files:
        img_path = os.path.join(dataset_info['images'], img_file)
        mask_file = img_file.replace('.jpg', '_mask.jpg')  # 또는 다른 확장자
        mask_path = os.path.join(dataset_info['masks'], mask_file)
        
        if os.path.exists(mask_path):
            ratio = calculate_mask_ratio(mask_path)
            if ratio <= 5.0:
                small_polyp_data.append({
                    'image': img_path,
                    'mask': mask_path,
                    'ratio': ratio,
                    'dataset': dataset_name
                })
    
    return small_polyp_data


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
        
    def visualize_feature_maps(self, feature_maps, module_name, save_path, original_image, ground_truth):
        if not isinstance(feature_maps, torch.Tensor):
            # tuple인 경우 첫 번째 텐서를 사용
            if isinstance(feature_maps, tuple):
                feature_maps = feature_maps[0]
            else:
                return

        print(f"\nModule: {module_name}")
        print(f"Feature map shape: {feature_maps.shape}")
        
        # 배치 차원이 있는 경우 첫 번째 배치만 사용
        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps[0]
        
        feature_maps = feature_maps.cpu().numpy()
        
        # Get original image size
        orig_h, orig_w = original_image.shape[:2]

        # Get attention flag
        is_attention = ('attention' in module_name.lower() or 
                    'aa_kernel' in module_name.lower() or 
                    'lablock' in module_name.lower() or 
                    'ma' in module_name.lower())

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
            
        # Feature map processing
        feature_map = np.mean(feature_maps, axis=0)
        
        # Normalize feature map values to 0-1 range
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Use threshold to find significant values (e.g., values > 1% of max)
        threshold = 0.01
        feature_map_binary = feature_map > threshold
        
        # Find the boundaries of actual content
        rows = np.any(feature_map_binary, axis=1)
        cols = np.any(feature_map_binary, axis=0)
        
        if np.any(rows) and np.any(cols):
            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]
            
            # Add small padding if needed (e.g., 5 pixels)
            pad = 5
            row_min = max(0, row_min - pad)
            row_max = min(feature_map.shape[0], row_max + pad)
            col_min = max(0, col_min - pad)
            col_max = min(feature_map.shape[1], col_max + pad)
            
            # Crop the feature map
            feature_map = feature_map[row_min:row_max, col_min:col_max]
        
        # Resize cropped feature map to match original image aspect ratio
        aspect_ratio = orig_w / orig_h
        new_h = int(np.sqrt(feature_map.size / aspect_ratio))
        new_w = int(new_h * aspect_ratio)
        feature_map_resized = cv2.resize(feature_map, (new_w, new_h))
        
        # Feature map visualization
        plt.subplot(1, 4, 3)
        if is_attention:
            plt.imshow(feature_map_resized, cmap='jet')
            plt.title(f'Attention Map\n({feature_maps.shape[0]} channels)')
        else:
            plt.imshow(feature_map_resized, cmap='jet')
            plt.title(f'Feature Map\n({feature_maps.shape[0]} channels)')
        plt.axis('off')
        
        # Overlay visualization
        plt.subplot(1, 4, 4)
        plt.imshow(original_image)
        feature_map_overlay = cv2.resize(feature_map_resized, (orig_w, orig_h))
        if is_attention:
            plt.imshow(feature_map_overlay, cmap='jet', alpha=0.6)
            plt.title('Attention Overlay')
        else:
            plt.imshow(feature_map_overlay, cmap='jet', alpha=0.5)
            plt.title('Feature Map Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Additional visualization for attention modules
        if is_attention and feature_maps.shape[0] > 1:
            plt.figure(figsize=(20, 4))
            num_heads = min(8, feature_maps.shape[0])
            
            for i in range(num_heads):
                plt.subplot(1, num_heads, i + 1)
                head_map = feature_maps[i]
                
                # Normalize and threshold
                head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
                head_map_binary = head_map > threshold
                
                # Find content boundaries
                rows = np.any(head_map_binary, axis=1)
                cols = np.any(head_map_binary, axis=0)
                
                if np.any(rows) and np.any(cols):
                    row_min, row_max = np.where(rows)[0][[0, -1]]
                    col_min, col_max = np.where(cols)[0][[0, -1]]
                    
                    # Add small padding
                    pad = 5
                    row_min = max(0, row_min - pad)
                    row_max = min(head_map.shape[0], row_max + pad)
                    col_min = max(0, col_min - pad)
                    col_max = min(head_map.shape[1], col_max + pad)
                    
                    # Crop the head map
                    head_map = head_map[row_min:row_max, col_min:col_max]
                
                # Resize maintaining aspect ratio
                aspect_ratio = orig_w / orig_h
                new_h = int(np.sqrt(head_map.size / aspect_ratio))
                new_w = int(new_h * aspect_ratio)
                head_map_resized = cv2.resize(head_map, (new_w, new_h))
                
                plt.imshow(original_image)
                head_map_overlay = cv2.resize(head_map_resized, (orig_w, orig_h))
                plt.imshow(head_map_overlay, cmap='jet', alpha=0.6)
                plt.title(f'Head {i+1}')
                plt.axis('off')
            
            plt.tight_layout()
            save_dir = os.path.dirname(save_path)
            plt.savefig(os.path.join(save_dir, f'{module_name}_heads.png'))
            plt.close()


def analyze_model(model, model_name, weight_path, data_item, dataset_name, output_dir):
    """모델 분석 및 시각화 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not verify_model(model, weight_path, device):
        print("모델 구조와 가중치가 일치하지 않습니다.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"Loaded weights from: {weight_path}")
    
    input_tensor, original_image = preprocess_image(data_item['image'])
    input_tensor = input_tensor.to(device)
    
    ground_truth = imread_kor(data_item['mask'], cv2.IMREAD_GRAYSCALE)
    
    image_name = os.path.splitext(os.path.basename(data_item['image']))[0]
    save_dir = os.path.join(
        output_dir,
        model_name,
        dataset_name,
        image_name
    )
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = FeatureMapVisualizer(model)
    model.eval()
    
    try:
        visualizer.register_hooks(MODEL_MODULES[model_name])
        
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):  # 만약 output이 tuple이라면 첫번째 요소를 사용
                output = output[0]
            output = torch.sigmoid(output)
            final_mask = (output > 0.5).float().squeeze().cpu().numpy()
        
        for name, feature in visualizer.features.items():
            print(f"\nVisualizing module: {name}")
            print(f"Feature shape: {feature.shape if isinstance(feature, torch.Tensor) else 'tuple'}")
            
            # 만약 feature가 tuple이라면 첫번째 텐서를 사용
            if isinstance(feature, tuple):
                feature = feature[0]
            
            save_path = os.path.join(save_dir, f'{name}_feature_maps.png')
            
            visualizer.visualize_feature_maps(
                feature,
                name,
                save_path,
                original_image,
                ground_truth
            )
        
        # Save final prediction visualization
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        im = plt.imshow(final_mask, cmap='jet')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Prediction')
        plt.title('Final Prediction')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(original_image)
        plt.imshow(final_mask, cmap='jet', alpha=0.5)
        plt.title('Prediction Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_prediction.png'), bbox_inches='tight', dpi=300)
        plt.close()
            
    finally:
        visualizer.remove_hooks()

def evaluate_model(model, dataset_items, device):
    """단일 모델에 대한 평가"""
    model.eval()
    results = []
    
    for data in dataset_items:
        input_tensor, _ = preprocess_image(data['image'])
        input_tensor = input_tensor.to(device)
        
        true_mask = imread_kor(data['mask'], cv2.IMREAD_GRAYSCALE)
        true_mask = (true_mask > 0).astype(np.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
        
        dice_score = Dice_Coefficient(
            torch.from_numpy(pred_mask), 
            torch.from_numpy(true_mask)
        )
        
        results.append({
            'image_path': data['image'],
            'dataset': data['dataset'],
            'polyp_ratio': data['ratio'],
            'dice_score': dice_score
        })
    
    return results

def evaluate_all_models_and_datasets():
    """각 모델의 데이터셋별 가중치를 사용하여 평가 수행"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import models
    from models.CaraNet import CaraNet
    from models.convsegnet import convsegnet
    from models.cps import cps
    from models.CFHA_Net import CFHA_Net
    from models.polyper import polyper
    
    # Initialize models dictionary
    models = {
        'CaraNet': lambda: CaraNet(in_channels=3, num_classes=1),
        'ConvSegNet': lambda: convsegnet(in_channels=3, number_of_classes=1),
        'CPS': lambda: cps(in_channels=3, out_channels=1),
        'CFHA': lambda: CFHA_Net(in_channel=3, out_channel=1),
        'Polyper': lambda: polyper(in_chans=3, num_classes=1)
    }
    
    all_results = []
    
    for dataset_name in DATASET_PATHS.keys():
        print(f"\nProcessing dataset: {dataset_name}")
        small_polyp_data = get_small_polyp_dataset(dataset_name)
        print(f"Found {len(small_polyp_data)} images with polyps <= 5% of image size")
        
        for model_name, model_init in models.items():
            print(f"\nEvaluating {model_name} on {dataset_name}")
            
            weight_path = WEIGHT_PATHS[model_name][dataset_name]
            if not os.path.exists(weight_path):
                print(f"Warning: Weight file not found at {weight_path}")
                continue
                
            try:
                model = model_init().to(device)
                model.load_state_dict(torch.load(weight_path, map_location=device))
                model.eval()
                
                print(f"Loaded weights from: {weight_path}")
                
                results = evaluate_model(model, small_polyp_data, device)
                
                for result in results:
                    result['model'] = model_name
                    result['weights_used'] = os.path.basename(weight_path)
                    all_results.append(result)
                
                print(f"Completed evaluation: Average Dice score = {np.mean([r['dice_score'] for r in results]):.3f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                continue
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    summary_df = pd.DataFrame(all_results)
    print("\nEvaluation Summary:")
    print("\nAverage Dice Scores by Model and Dataset:")
    avg_scores = summary_df.pivot_table(
        values='dice_score',
        index='model',
        columns='dataset',
        aggfunc='mean'
    )
    print(avg_scores)
    
    return all_results

def save_results_to_excel(results):
    """결과를 엑셀 파일로 저장하고 분석"""
    df = pd.DataFrame(results)
    
    df['poor_performance'] = df['dice_score'] < 0.5
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results_file = f"small_polyp_analysis_{timestamp}.xlsx"
    
    with pd.ExcelWriter(all_results_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        model_performance = df.groupby(['model', 'dataset'])['dice_score'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        model_performance.to_excel(writer, sheet_name='Model Performance')
        
        poor_cases = df[df['poor_performance']].sort_values('dice_score')
        poor_cases.to_excel(writer, sheet_name='Poor Performance Cases', index=False)
        
        dataset_stats = df.groupby('dataset').agg({
            'dice_score': ['count', 'mean', 'std', 'min', 'max'],
            'polyp_ratio': ['mean', 'min', 'max']
        }).round(3)
        dataset_stats.to_excel(writer, sheet_name='Dataset Statistics')
    
    print(f"\nResults saved to {all_results_file}")
    return df

def analyze_results(df):
    """결과 분석 및 요약 출력"""
    print("\n=== Analysis Summary ===")
    
    # 결과를 저장할 엑셀 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"analysis_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(analysis_file, engine='xlsxwriter') as writer:
        # Model Performance
        print("\nModel Performance:")
        model_performance = df.groupby('model')['dice_score'].agg(['mean', 'std', 'min', 'max'])
        print(model_performance)
        model_performance.to_excel(writer, sheet_name='Model Performance')
        
        # Dataset Performance
        print("\nDataset Performance:")
        dataset_performance = df.groupby('dataset')['dice_score'].agg(['mean', 'std', 'min', 'max'])
        print(dataset_performance)
        dataset_performance.to_excel(writer, sheet_name='Dataset Performance')
        
        # Polyp Size vs Performance
        print("\nPolyp Size vs Performance:")
        size_bins = pd.cut(df['polyp_ratio'], bins=[0, 1, 2, 3, 4, 5])
        size_performance = df.groupby(size_bins)['dice_score'].agg(['count', 'mean', 'std'])
        print(size_performance)
        size_performance.to_excel(writer, sheet_name='Size Performance')
        
        # Worst Performing Cases
        print("\nWorst Performing Cases (Dice < 0.3):")
        worst_cases = df[df['dice_score'] < 0.3].sort_values('dice_score')
        print(worst_cases[['model', 'dataset', 'image_path', 'polyp_ratio', 'dice_score']])
        worst_cases.to_excel(writer, sheet_name='Worst Cases')
        
        # All Results
        df.to_excel(writer, sheet_name='All Results', index=False)
    
    print(f"\nAnalysis results saved to: {analysis_file}")

# 7. Main Function
def main():
    try:
        # CPS 모델만 임포트
        from models.cps import cps
    except ImportError as e:
        print(f"모델 import 오류: {e}")
        print("models 디렉토리가 올바른 위치에 있는지 확인해주세요.")
        return

    # 시작 시간 기록
    start_time = datetime.now()
    print(f"Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 결과 저장 디렉토리 생성
    output_dir = f"feature_maps_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # CPS 모델만 초기화
    models = {
        'CPS': lambda: cps(in_channels=3, out_channels=1)
    }

    # 각 데이터셋에 대해 처리
    for dataset_name in DATASET_PATHS.keys():
        print(f"\nProcessing dataset: {dataset_name}")
        small_polyp_data = get_small_polyp_dataset(dataset_name)
        print(f"Found {len(small_polyp_data)} images with polyps <= 5% of image size")

        # CPS 모델에 대해서만 처리
        model_name = 'CPS'
        model_init = models[model_name]
        print(f"\nProcessing {model_name} on {dataset_name}")
        model = model_init()
        weight_path = WEIGHT_PATHS[model_name][dataset_name]

        # 각 이미지에 대해 처리
        for data_item in small_polyp_data:
            try:
                analyze_model(
                    model=model,
                    model_name=model_name,
                    weight_path=weight_path,
                    data_item=data_item,
                    dataset_name=dataset_name,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"Error processing {data_item['image']}: {str(e)}")
                continue

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 종료 시간 및 소요 시간 출력
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nAnalysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nResults saved in: {output_dir}")

if __name__ == '__main__':
    main()