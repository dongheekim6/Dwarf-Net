import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

from models.crc_real_v5 import crc_real_v5
from models.crc_test_v72 import crc_test_v72

# ====== 설정 ======
iter_num = int(input("📌 Iteration 번호를 입력하세요 (test_outputs 확인용): "))
data_type = 'kvasir_1000'
save_dir = '/userHome/userhome2/donghee/modelcombination/result_paper/'
os.makedirs(save_dir, exist_ok=True)

device_ids = [1, 3]  # 사용할 GPU 번호
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using devices: {device_ids}")
print(f"✅ Using device: {device}")

# ====== 경로 자동화 ======
def get_test_outputs_dir(model_name, iter_num):
    return f"/userHome/userhome2/donghee/modelcombination/_output/output_{data_type}/{model_name}_Iter_{iter_num}/test_outputs"

# ====== 이미지/GT 경로 ======
images_base_dir = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/images"
masks_base_dir  = "/userHome/userhome2/donghee/modelcombination/Dataset_processing/Kvasir-SEG/masks"

# ====== feature map 시각화 함수 ======
def normalize_feature_map(fm):
    fm = fm.detach().cpu().numpy()
    fm = np.mean(fm, axis=0)  # 채널 평균
    fm -= fm.min()
    fm /= (fm.max() + 1e-8)
    fm = (fm * 255).astype(np.uint8)
    return cv2.applyColorMap(fm, cv2.COLORMAP_JET)

def visualize_decoder_outputs(model, input_tensor, model_name, img_name):
    model.eval()
    decoder_outputs = []
    hooks = []

    def hook_fn(module, input, output):
        decoder_outputs.append(output[0])  # batch=0

    if hasattr(model, 'decoders'):  # crc 모델
        for decoder in model.decoders:
            hooks.append(decoder.register_forward_hook(hook_fn))
    elif hasattr(model, 'decoder1'):  # 제안모델
        hooks.append(model.decoder1.register_forward_hook(hook_fn))
    else:
        raise RuntimeError("디코더 모듈을 찾을 수 없습니다.")

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    # 시각화
    n_cols = len(decoder_outputs) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))

    inp = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(inp)
    axes[0].set_title("Input")
    axes[0].axis('off')

    for i, fm in enumerate(decoder_outputs):
        heatmap = normalize_feature_map(fm)
        axes[i+1].imshow(heatmap)
        axes[i+1].set_title(f'Decoder{i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{img_name}_{model_name}_decoder_outputs.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

# ====== 입력 이미지 로드 ======
def load_image(path, size=(352, 352)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0).to(device)

# ====== test_outputs 에서 이미지 찾기 ======
def get_image_name_from_test_outputs(test_outputs_dir):
    npy_files = sorted(Path(test_outputs_dir).glob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy found in {test_outputs_dir}")
    return npy_files[0].stem  # 예: cju123.png

def find_original_image(img_name, base_dir):
    base = img_name.rsplit('.', 1)[0]
    for ext in ['.jpg', '.png', '.jpeg', '.tif']:
        path = os.path.join(base_dir, base+ext)
        if os.path.exists(path):
            return path
    raise RuntimeError(f"원본 이미지 없음: {img_name}")

# ====== 메인 ======
for model_name, model_class in zip(['crc_real_v5', 'crc_test_v72'], [crc_real_v5, crc_test_v72]):
    print(f"\n🔷 {model_name} 처리중...")

    weight_path = input(f"💾 {model_name}의 가중치 경로를 입력하세요: ").strip()
    test_outputs_dir = get_test_outputs_dir(model_name, iter_num)

    print(f"  🔹 가중치: {weight_path}")
    print(f"  🔹 test_outputs: {test_outputs_dir}")

    if not os.path.exists(weight_path):
        print(f"❌ 가중치 파일이 없습니다: {weight_path}")
        continue
    if not os.path.exists(test_outputs_dir):
        print(f"❌ test_outputs 디렉토리가 없습니다: {test_outputs_dir}")
        continue

    # test_outputs 에서 첫 번째 이미지 선택
    img_name = get_image_name_from_test_outputs(test_outputs_dir)
    print(f"  🔹 선택된 이미지: {img_name}")

    # 원본 이미지 경로 찾기
    img_path = find_original_image(img_name, images_base_dir)
    print(f"  🔹 원본 이미지: {img_path}")

    # 모델 로드
    model = model_class().to(device)
    state_dict = torch.load(weight_path, map_location=device)

    # 'module.' prefix가 붙어 있으면 제거
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    print("  ✅ 모델 로드 완료")

    # 입력 이미지 로드
    input_tensor = load_image(img_path)

    # 디코더 출력 시각화
    visualize_decoder_outputs(model, input_tensor, model_name, img_name)