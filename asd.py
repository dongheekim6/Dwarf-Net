import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from collections import OrderedDict
from models.crc_test_v61 import crc_test_v61  # 경로 맞는지 확인

def calculate_metrics(tp_sum, gt_pixels):
    tp_rate = tp_sum / gt_pixels if gt_pixels > 0 else 0
    return tp_sum, tp_rate

def analyze_encoder_features(model, image, mask, device):
    model.to(device)
    model.eval()
    image = image.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        e1 = model.encoder1(image)
        e2 = model.encoder2(e1)
        e3 = model.encoder3(e2)
        e4 = model.encoder4(e3)

        encoder_feats = [e1, e2, e3, e4]
        results = {}

        # 마스크 이진화 후 픽셀 수
        mask_bin = (mask > 0.5).float()  # (B, 1, H, W)
        gt_pixels = torch.sum(mask_bin).item()

        for idx, feat in enumerate(encoder_feats):
            feat_up = F.interpolate(feat, size=mask.shape[-2:], mode='bilinear', align_corners=True)
            feat_mean = torch.mean(feat_up, dim=1, keepdim=True)  # (B, 1, H, W)

            # 정규화
            feat_norm = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)

            tp_value = torch.sum(feat_norm * mask_bin).item()
            tp_value, tp_rate = calculate_metrics(tp_value, gt_pixels)

            results[f"encoder{idx+1}"] = {
                "TP": tp_value,
                "GT Pixels": gt_pixels,
                "TP Rate (TP / GT)": tp_rate
            }

    return results

def load_image_and_mask(image_path, mask_path, image_size=(256, 256)):
    image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    image = cv2.resize(image, image_size)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, image_size)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    mask_tensor = transform(mask).unsqueeze(0)    # (1, 1, H, W)

    return image_tensor, mask_tensor

def main():
    image_path = input("분석할 이미지 경로를 입력해주세요: ").strip()
    mask_path = input("마스크 이미지 경로를 입력해주세요: ").strip()
    weights_path = input("학습된 모델 가중치(.pth) 경로를 입력해주세요: ").strip()

    device = 'cpu'  # 강제로 CPU 사용
    print(f"\n[INFO] 사용 디바이스: {device.upper()}")

    model = crc_test_v61(in_channels=3, num_classes=1)

    # 모델 가중치 로드
    state_dict = torch.load(weights_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)

    # 이미지 및 마스크 로드
    image_tensor, mask_tensor = load_image_and_mask(image_path, mask_path)

    # 인코더 출력 분석
    results = analyze_encoder_features(model, image_tensor, mask_tensor, device=device)

    # 결과 출력
    print("\n[분석 결과]")
    for layer, metrics in results.items():
        print(f"[{layer}] TP: {metrics['TP']:.2f} / GT Pixels: {metrics['GT Pixels']:.0f} → TP Rate: {metrics['TP Rate (TP / GT)']:.4f}")

if __name__ == '__main__':
    main()
