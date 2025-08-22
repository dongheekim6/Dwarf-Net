import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from skimage.transform import resize

import sys
sys.path.append('/userHome/userhome2/donghee/modelcombination')

# Models
from models.crc_test_v72 import crc_test_v72
from models.crc_real_v5 import crc_real_v5
from models.CaraNet import CaraNet
from models.convsegnet import ConvSegNet
from models.cps import cps
from models.CFHA_Net import CFHA_Net
from models.polyper import polyper

# Settings
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Utility

def load_image(path, gray=False, size=(352,352)):
    img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # 원본 이미지를 컬러로 보존하기 위해 별도로 저장
    if len(img_cv.shape) == 2:
        img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    else:
        img_cv_color = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv_color = cv2.resize(img_cv_color, size)
    
    if gray:
        if len(img_cv.shape) == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_cv = cv2.resize(img_cv, size)
        img_np = img_cv
        img = Image.fromarray(img_cv)
        tensor = transform_gray(img).unsqueeze(0).to(device)
    else:
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = cv2.resize(img_cv, size)
        img_np = img_cv
        img = Image.fromarray(img_cv)
        tensor = transform_rgb(img).unsqueeze(0).to(device)
    
    # 컬러 이미지를 반환하도록 수정
    return tensor, img_cv_color


def load_gt(path, size=(352,352)):
    gt = Image.open(path).convert('L').resize(size)
    return np.array(gt) / 255.0


def load_model(cls, weight, in_channels=3, num_classes=1):
    try:
        model = cls(num_classes=num_classes, in_channels=in_channels)
    except TypeError:
        try:
            model = cls(num_classes=num_classes)
        except TypeError:
            model = cls()

    state = torch.load(weight, map_location=device, weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.',''):v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def predict(model, x, shape):
    with torch.no_grad():
        out = model(x)
        if out.shape[1] == 1:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=1)
        pred = out.cpu().numpy()[0,0]
        if pred.shape != shape:
            pred = resize(pred, shape, order=1, mode='reflect', anti_aliasing=False)
    return pred


def dice(pred, gt):
    pred_bin = (pred>0.5).astype(np.float32)
    gt_bin = (gt>0.5).astype(np.float32)
    inter = np.sum(pred_bin*gt_bin)
    union = np.sum(pred_bin)+np.sum(gt_bin)
    if union==0: return 1.0 if np.sum(pred_bin)==0 else 0.0
    return 2*inter/union

# Config

models_config = {
    'Proposed': {
        'class': crc_test_v72,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250630_202335/crc_test_v72_Iter_10/250701_102020_crc_test_v72_Iter_10.pt',
        'in_channels':3
    },
    'crc_real_v5':{
        'class': crc_real_v5,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250528_045325/crc_real_v5_Iter_10/250528_142644_crc_real_v5_Iter_10.pt',
        'in_channels':3
    },
    'CaraNet':{
        'class': CaraNet,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_CaraNet_kvasir/output_250213_193358/CaraNet_Iter_1/250213_193400_CaraNet_Iter_1.pt',
        'in_channels':3
    },
    'convsegnet':{
        'class': ConvSegNet,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_convsegnet_kvasir/output_250221_010415/convsegnet_Iter_2/250221_022925_convsegnet_Iter_2.pt',
        'in_channels': 3
    },
    'cps':{
        'class': cps,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_cps_kvasir/output_250221_100654/cps_Iter_2/250221_113135_cps_Iter_2.pt',
        'in_channels':3
    },
    'CFHA_Net':{
        'class': CFHA_Net,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_CFHA_Net_kvasir/output_250221_022623/CFHA_Net_Iter_1/250221_022624_CFHA_Net_Iter_1.pt',
        'in_channels':3
    },
    'polyper':{
        'class': polyper,
        'weight': '/userHome/userhome2/donghee/modelcombination/output_polyper_kvasir/output_250224_012842/polyper_Iter_1/250224_012843_polyper_Iter_1.pt',
        'in_channels':3
    }
}

images = [
    '/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/images/cjyzuio1qgh040763k56deohv.jpg',
    '/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/images/cju5uget8krjy0818kvywd0zu.jpg',
    '/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/images/cju2t2ivz43i10878zeg8r1br.jpg'
]

ground_truths = [
    '/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/masks/cjyzuio1qgh040763k56deohv.jpg',
    '/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/masks/cju5uget8krjy0818kvywd0zu.jpg',
    '/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/masks/cju2t2ivz43i10878zeg8r1br.jpg'
]

# Load models
loaded_models = {}
for name,cfg in models_config.items():
    print(f"Loading {name}...")
    try:
        loaded_models[name] = load_model(cfg['class'], cfg['weight'],
                                         in_channels=cfg['in_channels'])
    except Exception as e:
        print(f"Failed: {e}")
        loaded_models[name]=None

# Evaluate

results = []

for idx,img_path in enumerate(images):
    print(f"\nProcessing {img_path}")
    gt = load_gt(ground_truths[idx])
    shape = gt.shape
    preds = {}
    dscs = {}
    
    for name,model in loaded_models.items():
        if model is None:
            preds[name]=np.zeros(shape)
            dscs[name]=0.0
            continue
        gray = False
        img_tensor, img_np = load_image(img_path, gray=gray)
        pred = predict(model,img_tensor,shape)
        dsc = dice(pred,gt)
        preds[name]=pred
        dscs[name]=dsc
        print(f"  {name}: DSC={dsc:.4f}")
    
    results.append({
        'img':img_np, 'gt':gt,
        'preds':preds, 'dscs':dscs
    })

# Visualization with improved layout

n_models = len(models_config)+2
fig, axes = plt.subplots(len(images), n_models, figsize=(3*n_models, 6*len(images)))

if len(images)==1:
    axes = np.expand_dims(axes,0)

# 첫 번째 행에만 컬럼 제목 추가 (더 큰 폰트)
column_titles = ["Input", "Ground Truth"] + list(models_config.keys())

for col, title in enumerate(column_titles):
    if title == "Proposed":
        # Proposed는 빨간색으로 표시
        axes[0, col].text(0.5, 1.15, title, 
                         transform=axes[0, col].transAxes,
                         ha='center', va='center', 
                         fontsize=20, fontweight='bold', color='red')
    else:
        axes[0, col].text(0.5, 1.15, title, 
                         transform=axes[0, col].transAxes,
                         ha='center', va='center', 
                         fontsize=20, fontweight='bold')

for row,res in enumerate(results):
    img_show = res['img']
    
    # 이미지가 이미 컬러이므로 추가 처리 불필요
    # 픽셀 값을 0-255 범위로 확인하고 정규화
    if img_show.max() <= 1.0:
        img_show = (img_show * 255).astype(np.uint8)
    else:
        img_show = img_show.astype(np.uint8)
    
    # Input 이미지
    axes[row,0].imshow(img_show)
    axes[row,0].axis('off')
    
    # Input 이미지 아래에 이미지 이름 표시 제거
    # img_name = os.path.basename(images[row]).split('.')[0]
    # axes[row,0].text(0.5, -0.08, img_name, 
    #                 transform=axes[row,0].transAxes,
    #                 ha='center', va='top', 
    #                 fontsize=12, fontweight='bold')

    # Ground Truth
    axes[row,1].imshow(res['gt'],cmap='gray')
    axes[row,1].axis('off')

    # 모델 예측 결과들
    for col,(name,pred) in enumerate(res['preds'].items()):
        axes[row,col+2].imshow(pred,cmap='gray')
        axes[row,col+2].axis('off')
        
        # 각 이미지 아래에 DSC 점수 표시 (더 큰 폰트)
        if name == 'Proposed':
            # Proposed는 빨간색으로 표시
            axes[row,col+2].text(0.5, -0.08, f"DSC: {res['dscs'][name]:.4f}", 
                                transform=axes[row,col+2].transAxes,
                                ha='center', va='top', 
                                fontsize=18, fontweight='bold', color='red')
        else:
            axes[row,col+2].text(0.5, -0.08, f"DSC: {res['dscs'][name]:.4f}", 
                                transform=axes[row,col+2].transAxes,
                                ha='center', va='top', 
                                fontsize=18, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)  # 행 간격 조정
plt.savefig("comparison_fixed.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved comparison_fixed.png")