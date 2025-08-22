import os, glob
import numpy as np
import cv2
from pathlib import Path
from models.crc_test_v72 import crc_test_v72
from models.crc_real_v5 import crc_real_v5
from models.CaraNet import CaraNet
from models.convsegnet import ConvSegNet
from models.CFHA_Net import CFHA_Net
from models.polyper import polyper

device = 'cuda:0'

# 데이터셋 & 결과 경로
images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/ETIS-LaribPolypDB/images"
masks_base_dir  = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/ETIS-LaribPolypDB/masks"

iter = 1
data_type = 'etis'

models_config = {
    # 'Proposed': {
    #     'class': crc_test_v72,
    #     'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/crc_test_v72_Iter_{iter}/test_outputs',
    # },
    'CRCNet':{
        'class': crc_real_v5,
        'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/crc_real_v5_Iter_{iter}/test_outputs',
    }
    # 'CaraNet':{
    #     'class': CaraNet,
    #     'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/CaraNet_Iter_{iter}/test_outputs',
    # },
    # 'ConvSegNet':{
    #     'class': ConvSegNet,
    #     'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/convsegnet_v2_Iter_{iter}/test_outputs',
    # },
    # 'CFHA_Net':{
    #     'class': CFHA_Net,
    #     'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/CFHA_Net_Iter_{iter}/test_outputs',
    # },
    # 'Polyper':{
    #     'class': polyper,
    #     'test_outputs_dir': f'/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/polyper_Iter_{iter}/test_outputs',
    # }
}
def find_gt(name, mask_dir):
    base = Path(name).stem
    digits = ''.join(filter(str.isdigit, base))
    for f in os.listdir(mask_dir):
        f_base = Path(f).stem
        if digits == ''.join(filter(str.isdigit, f_base)):
            return os.path.join(mask_dir, f)
    return None
def load_mask(path, size=(352,352)):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, size) / 255.0
    return gt

def dice(pred, gt):
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    inter = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    return 1.0 if union == 0 else 2.0 * inter / union

def iou(pred, gt):
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    inter = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - inter
    return 1.0 if union == 0 else inter / union

def load_test_outputs(output_dir):
    outputs = {}
    for npy_path in glob.glob(os.path.join(output_dir, '*.npy')):
        name = Path(npy_path).stem
        pred = np.load(npy_path).squeeze()
        if pred.ndim == 3:
            pred = pred[0]
        if pred.max() > 1.0:
            pred = pred / 255.0
        outputs[name] = pred
    return outputs

print("\n=== Evaluating Models ===")
for model_name, config in models_config.items():
    output_dir = config['test_outputs_dir']
    print(f"\nModel: {model_name}")
    if not os.path.exists(output_dir):
        print(f"  ⚠️ Output directory not found: {output_dir}")
        continue
    
    preds = load_test_outputs(output_dir)
    if not preds:
        print("  ⚠️ No predictions found.")
        continue

    dices, ious = [], []
    for name, pred in preds.items():
        # GT 찾기
        gt_path = find_gt(name, masks_base_dir)
        if gt_path is None:
            print(f"  ⚠️ GT not found for {name}")
            continue
        
        gt = load_mask(gt_path, size=pred.shape[::-1])
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        dices.append(dice(pred, gt))
        ious.append(iou(pred, gt))
    
    if dices:
        print(f"  Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"  IoU : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    else:
        print("  ⚠️ No valid results.")