import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.CaraNet import CaraNet
from models.convsegnet_v2 import convsegnet_v2
from models.cpsformer import cpsformer
from models.CFHA_Net import CFHA_Net
from models.polyper import polyper
from models.crc_test_v72 import crc_test_v72
from models.crc_real_v5 import crc_real_v5
from tqdm import tqdm
import pandas as pd
from monai.metrics import compute_iou as IoU_Function

def imread_kor(filePath, mode=cv2.IMREAD_UNCHANGED):
    stream = open(filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray, mode)
def iou_score(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
def preprocess_image(image_path):
    image_bgr = imread_kor(image_path, mode=cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(image_rgb).float()
    
    if tensor.shape[-2:] != (352, 352):
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(352, 352),
            mode='bilinear', align_corners=False
        ).squeeze(0)
    
    return tensor.unsqueeze(0)

def preprocess_mask(mask_path):
    mask = imread_kor(mask_path, mode=cv2.IMREAD_GRAYSCALE)
    
    transform = transforms.ToTensor()
    tensor = transform(mask).float()
    
    if tensor.shape[-2:] != (352, 352):
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(352, 352),
            mode='nearest'
        ).squeeze(0)
    
    tensor[tensor > 0] = 1
    
    return tensor.unsqueeze(0)

def remove_module_prefix(state_dict):
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
def compute_metrics(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }
def dice_coefficient(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def Confusion_Matrix(yhat, ytrue, threshold=0.5):
    yhat = (yhat > threshold).cpu().numpy()
    ytrue = (ytrue > threshold).cpu().numpy()

    tp = np.sum(yhat * ytrue)
    fp = np.sum(yhat * (1 - ytrue))
    fn = np.sum((1 - yhat) * ytrue)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return recall, precision, f1

def evaluate_model_split_level(model, weight_path, image_paths, mask_paths, device):
    if not image_paths or not mask_paths:
        print("⚠️ No images found!")
        return {}

    model.load_state_dict(remove_module_prefix(torch.load(weight_path, map_location=device, weights_only=True)))
    model.eval()

    outputs_all = []
    targets_all = []
    dice_per_image = []

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        try:
            image = preprocess_image(img_path).to(device)
            mask = preprocess_mask(mask_path).to(device)
        except Exception as e:
            print(f"⚠️ Error processing:\nImage: {img_path}\nMask: {mask_path}\nError: {str(e)}")
            continue

        with torch.no_grad():
            output = model(image)
            output = torch.sigmoid(output)

        pred = output.squeeze(0).squeeze(0).cpu()
        gt = mask.squeeze(0).squeeze(0).cpu()

        outputs_all.append(pred.unsqueeze(0))
        targets_all.append(gt.unsqueeze(0))

        d = dice_coefficient(pred, gt)
        dice_per_image.append(d)

    # 전체 합치기
    if not outputs_all:
        return {}

    outputs_all = torch.cat(outputs_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    # threshold 적용
    outputs_bin = (outputs_all > 0.5).float()

    # Dice 및 기타 지표 계산
    dice = dice_coefficient(outputs_bin, targets_all)
    iou = iou_score(outputs_bin, targets_all)
    recall, precision, f1 = Confusion_Matrix(outputs_bin, targets_all)
    dice_by_thresh = {"0~5%": [], "0~10%": [], "0~25%": [], "0~50%": []}
    iou_by_thresh = {"0~5%": [], "0~10%": [], "0~25%": [], "0~50%": []}
    prec_by_thresh = {"0~5%": [], "0~10%": [], "0~25%": [], "0~50%": []}
    rec_by_thresh = {"0~5%": [], "0~10%": [], "0~25%": [], "0~50%": []}

    ratio_bin_counts = {"0~5%": 0, "0~10%": 0, "0~25%": 0, "0~50%": 0}

    for pred, gt in zip(outputs_all, targets_all):
        ratio = gt.sum() / (gt.shape[-2] * gt.shape[-1])  # 마스크 내 1의 비율
        ratio = ratio.item()

        if ratio <= 0.05:
            label = "0~5%"
        elif ratio <= 0.10:
            label = "0~10%"
        elif ratio <= 0.25:
            label = "0~25%"
        elif ratio <= 0.50:
            label = "0~50%"
        else:
            continue  # 50% 초과는 제외

        ratio_bin_counts[label] += 1

        metrics = compute_metrics((pred > 0.5).float(), gt)
        dice_by_thresh[label].append(metrics['dice'])
        iou_by_thresh[label].append(metrics['iou'])
        prec_by_thresh[label].append(metrics['precision'])
        rec_by_thresh[label].append(metrics['recall'])

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_images': outputs_all.shape[0],
        'dice_per_image': dice_per_image,
        'dice_by_thresh': dice_by_thresh,
        'iou_by_thresh': iou_by_thresh,
        'prec_by_thresh': prec_by_thresh,
        'rec_by_thresh': rec_by_thresh,
        'ratio_bin_counts': ratio_bin_counts
    }
def evaluate_dataset_split_level(model, weight_path, data_dir, device, dataset_name=""):
    image_paths = []
    mask_paths = []

    jpg_images = glob.glob(os.path.join(data_dir, "*[!_mask].jpg"))
    for img_path in jpg_images:
        mask_path = img_path.replace(".jpg", "_mask.jpg")
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)

    png_images = glob.glob(os.path.join(data_dir, "*[!_mask].png"))
    for img_path in png_images:
        mask_path = img_path.replace(".png", "_mask.png")
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)

    if not image_paths:
        print(f"⚠️ No valid image-mask pairs found in {data_dir}")
        return None

    print(f"Found {len(image_paths)} {dataset_name} image-mask pairs")

    try:
        results = evaluate_model_split_level(model, weight_path, image_paths, mask_paths, device)
        return results
    except Exception as e:
        print(f"⚠️ Error evaluating {dataset_name} set: {str(e)}")
        return None
# if __name__ == "__main__":
#     weight_root = "/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250528_045325"
#     dataset_dir = "/userHome/userhome2/donghee/modelcombination/splits_kvasir_1000"
# if __name__ == "__main__":
#     weight_root = "/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250528_045325"
#     dataset_dir = "/userHome/userhome2/donghee/modelcombination/splits_kvasir_1000"
# if __name__ == "__main__":
#     weight_root = "/userHome/userhome2/donghee/modelcombination/output_crc_test_dense/output_250528_045325"
#     dataset_dir = "/userHome/userhome2/donghee/modelcombination/splits_kvasir_1000"

if __name__ == "__main__":
    weight_root = "/userHome/userhome2/donghee/modelcombination/_output/output_etis"
    dataset_dir = "/userHome/userhome2/donghee/modelcombination/splits_Etis"
    
    
    # model = polyper(in_channels=3, out_channels=1)
    # model = polyper(in_channel=3, out_channel=1)
    # model = polyper(in_channels=3, number_of_classes=1)
    # model = cpsformer(in_channels=3, num_classes=1)
    model = crc_test_v72(in_channels=3, num_classes=1)
    devices = [3]  # 사용할 CUDA 디바이스 번호

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{devices[0]}")
    else:
        device = torch.device("cpu")

    model.to(device)

    results_data = []
    all_split_dices = []
    all_dice_per_image = []

    ratio_image_counts_total = {label: 0 for label in ["0~5%", "0~10%", "0~25%", "0~50%"]}

    for i in range(1, 11):
        split_dir = os.path.join(dataset_dir, f"split{str(i).zfill(2)}")
        if not os.path.exists(split_dir):
            print(f"⚠️ Split directory not found: {split_dir}")
            continue

        print(f"\n========== Processing split {i} ==========")

        subdir = os.path.join(weight_root, f"crc_test_v72_Iter_{i}")
        pt_files = glob.glob(os.path.join(subdir, "*.pt"))
        if not pt_files:
            print(f"⚠️ No .pt file found in {subdir}")
            continue

        pt_path = pt_files[0]
        print(f"✅ Using weights: {os.path.basename(pt_path)}")

        split_results = {}
        split_all_dices = []

        for dataset_type in ['test']:
            data_dir = os.path.join(split_dir, dataset_type)
            if not os.path.exists(data_dir):
                print(f"⚠️ {dataset_type} directory not found: {data_dir}")
                continue

            print(f"\nEvaluating {dataset_type} set:")
            try:
                results = evaluate_dataset_split_level(model, pt_path, data_dir, device, dataset_type)
                if results is not None:
                    dice = results['dice']
                    iou = results['iou']
                    precision = results['precision']
                    recall = results['recall']
                    f1 = results['f1']
                    num_images = results['num_images']

                    # ✅ ratio_bin_counts 추가 로그에 등록
                    ratio_bin_counts = results.get('ratio_bin_counts', {})
                    for label in ratio_bin_counts:
                        ratio_image_counts_total[label] += ratio_bin_counts[label]

                    all_dice_per_image.extend(results['dice_per_image'])
                    dice_by_thresh = results['dice_by_thresh']
                    iou_by_thresh = results['iou_by_thresh']
                    prec_by_thresh = results['prec_by_thresh']
                    rec_by_thresh = results['rec_by_thresh']

                    print(f"\n{dataset_type.capitalize()} Set:")
                    print(f"Dice 평균: {dice:.4f}")
                    print(f"IoU 평균: {iou:.4f}")
                    print(f"Precision 평균: {precision:.4f}")
                    print(f"Recall 평균: {recall:.4f}")
                    print(f"F1 Score: {f1:.4f}")

                    split_results[f"{dataset_type}_dice"] = dice
                    split_results[f"{dataset_type}_iou"] = iou
                    split_results[f"{dataset_type}_precision"] = precision
                    split_results[f"{dataset_type}_recall"] = recall
                    split_results[f"{dataset_type}_f1"] = f1
                    split_results[f"{dataset_type}_num_images"] = num_images

                    for label in ["0~5%", "0~10%", "0~25%", "0~50%"]:
                        if label in dice_by_thresh and dice_by_thresh[label]:
                            split_results[f"{dataset_type}_dice_thresh_{label}"] = float(np.mean(dice_by_thresh[label]))
                            split_results[f"{dataset_type}_iou_thresh_{label}"] = float(np.mean(iou_by_thresh[label]))
                            split_results[f"{dataset_type}_prec_thresh_{label}"] = float(np.mean(prec_by_thresh[label]))
                            split_results[f"{dataset_type}_rec_thresh_{label}"] = float(np.mean(rec_by_thresh[label]))

            except Exception as e:
                print(f"⚠️ Error evaluating {dataset_type} set: {str(e)}")
                continue

        if split_results:
            if "test_dice" in split_results:
                all_split_dices.append(split_results["test_dice"])
                print(f"Split {i} test Dice: {split_results['test_dice']:.4f}")

            split_results['split'] = i
            split_results['overall_dice'] = split_results["test_dice"]
            results_data.append(split_results)

    if not results_data:
        print("⚠️ No results were collected from any split!")
        exit(1)

    print("\n========== 🎯 최종 결과 ==========")

    if all_split_dices:
        final_mean = np.mean(all_split_dices)
        final_std = np.std(all_split_dices)
        print(f"전체 실험의 Dice 평균 (split 단위): {final_mean:.4f} ± {final_std:.4f}")

    # 평균값 계산을 위한 값만 계산 (시간상 중요치 없음)
    results_df = pd.DataFrame(results_data)

    for label in ["0~5%", "0~10%", "0~25%", "0~50%"]:
        # 각 split별 성능을 수집
        dice_col = f"test_dice_thresh_{label}"
        iou_col = f"test_iou_thresh_{label}"
        prec_col = f"test_prec_thresh_{label}"
        rec_col = f"test_rec_thresh_{label}"
        
        if dice_col in results_df.columns:
            dice_values = results_df[dice_col].dropna()
            iou_values = results_df[iou_col].dropna()
            prec_values = results_df[prec_col].dropna()
            rec_values = results_df[rec_col].dropna()
            
            if not dice_values.empty:
                print(f"\n용종 비율 {label} 성능:")
                print(f"  Dice 평균:      {dice_values.mean():.4f} ± {dice_values.std():.4f}")
                print(f"  IoU 평균:       {iou_values.mean():.4f} ± {iou_values.std():.4f}")
                print(f"  Precision 평균: {prec_values.mean():.4f} ± {prec_values.std():.4f}")
                print(f"  Recall 평균:    {rec_values.mean():.4f} ± {rec_values.std():.4f}")

    print(f"\n전체 이미지 수: {sum(ratio_image_counts_total.values())}")
    print("비율별 이미지 분포:")
    for label, count in ratio_image_counts_total.items():
        percentage = (count / sum(ratio_image_counts_total.values())) * 100 if sum(ratio_image_counts_total.values()) > 0 else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")