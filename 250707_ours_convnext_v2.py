#modified from 240714_main_(2D_add_slice).ipynb
import numpy as np
import pandas as pd
import time
import timeit
from datetime import datetime
import os
import glob
import natsort
import sys
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import cv2
from PIL import Image
import random
import copy
import warnings
warnings.filterwarnings('ignore')
# import ipynbname
if '__file__' in globals():
    FILENAME = os.getcwd() + '/' + os.path.basename(__file__)
else:
    try:
        from ipynbname import name
        FILENAME = os.getcwd() + '/' + name() + '.ipynb'
    except ModuleNotFoundError:
        FILENAME = os.getcwd() + '/notebook_name.ipynb'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, KFold

from monai.losses import TverskyLoss as TverskyLoss
from monai.transforms import Compose, ToTensor, RandFlip
from monai.metrics import DiceMetric as Dice_Function
from monai.metrics import compute_iou as IoU_Function
from monai.metrics import ConfusionMatrixMetric

import sys
sys.path.append("..")

def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False
def imread_kor ( filePath, mode=cv2.IMREAD_UNCHANGED ) : 
    stream = open( filePath.encode("utf-8") , "rb") 
    bytes = bytearray(stream.read()) 
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , mode)

def imwrite_kor(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result:
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
                return True
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False
    
def random_rotation(image, mask, angle_range=(-30, 30)):
    # 지정된 각도 범위 내에서 무작위로 각도 선택
    angle = random.uniform(angle_range[0], angle_range[1])
    # 이미지와 마스크를 동일한 각도로 회전
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    return image, mask

class ImagesDataset(Dataset):
    def __init__(self, image_path_list, target_path_list, aug=False):
        self.image_path_list = image_path_list
        self.target_path_list = target_path_list
        
        # (A) 원본 이미지를 위한 Transform (3채널)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),   # (H, W, 3) → (3, H, W)
        ])
        # (B) 마스크를 위한 Transform (1채널)
        self.mask_transform = transforms.ToTensor()  # (H, W) → (1, H, W)
        
        self.aug = aug  # 필요하면 사용

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        mask_path = self.target_path_list[idx]
        
        # 1. 원본 이미지는 3채널(BGR)로 강제 로드
        image_bgr = imread_kor(image_path, mode=cv2.IMREAD_COLOR)  
        # 2. BGR → RGB 변환
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)         # (H, W, 3)

        # 3. 마스크는 1채널(Grayscale)로 강제 로드
        mask = imread_kor(mask_path, mode=cv2.IMREAD_GRAYSCALE)    # (H, W)

        # 4. ToTensor 변환
        image = self.image_transform(image).float()  # shape: [3, H, W]
        mask = self.mask_transform(mask).float()     # shape: [1, H, W]

        # 5. (선택) 352×352로 리사이즈 (bilinear, nearest)
        if image.shape[-2:] != (352, 352):
            image = F.interpolate(
                image.unsqueeze(0), size=(352, 352),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if mask.shape[-2:] != (352, 352):
            mask = F.interpolate(
                mask.unsqueeze(0), size=(352, 352),
                mode='nearest'
            ).squeeze(0)

        # 6. 이진화 (마스크 값이 0보다 크면 1로)
        mask[mask > 0] = 1
        
        return image, mask, image_path
def Pixel_Accuracy(yhat, ytrue, threshold=0.5):
    yhat = yhat>threshold
    correct = torch.sum(yhat == ytrue)
    total = ytrue.numel()
    accuracy = correct.float() / total
    return accuracy.item()

def Intersection_over_Union(yhat, ytrue, threshold=0.5):
    yhat = yhat>threshold
    return IoU_Function(yhat, ytrue).nanmean().item()
 
def Dice_Coefficient(yhat, ytrue, threshold=0.5):
    yhat = (yhat > threshold).cpu().numpy()
    ytrue = (ytrue > threshold).cpu().numpy()

    intersection = np.sum(yhat * ytrue)
    union = np.sum(yhat) + np.sum(ytrue)

    dice = (2 * intersection) / (union + 1e-6)
    return dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # sigmoid 제거: 이미 외부에서 처리됨
        # inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    
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

def train(train_loader, epoch, model, criterion, optimizer, device):
    model.train()
    train_losses = AverageMeter()
    
    for i, (input, target, _) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        
        # Get the output
        output = model(input)
        output = nn.Sigmoid()(output)
        
        # Calculate loss
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.update(loss.detach().cpu().numpy(), input.shape[0])
    
    Train_Loss = np.round(train_losses.avg, 6)
    return Train_Loss

def validate(validation_loader, model, criterion, device, model_path=False, return_image_paths=False):
    if model_path != False:
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    
    for i, (input, target, image_path) in enumerate(validation_loader):
        input = input.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            # Get the output
            output = model(input)
            output = nn.Sigmoid()(output)
            
        if i == 0:
            targets = target
            outputs = output
            if return_image_paths:
                image_paths = image_path
        else:
            targets = torch.cat((targets, target))
            outputs = torch.cat((outputs, output), axis=0)
            if return_image_paths:
                image_paths += image_path
                
    if return_image_paths:
        return outputs, targets, image_paths
    return outputs, targets


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def copy_sourcefile(output_dir, src_dir = 'src' ):    
    import os 
    import shutil
    import glob 
    source_dir = os.path.join(output_dir, src_dir)

    os.makedirs(source_dir, exist_ok=True)
    org_files1 = os.path.join('./', '*.py' )
    org_files2 = os.path.join('./', '*.sh' )
    org_files3 = os.path.join('./', '*.ipynb' )
    org_files4 = os.path.join('./', '*.txt' )
    org_files5 = os.path.join('./', '*.json' )    
    files =[]
    files = glob.glob(org_files1 )
    files += glob.glob(org_files2  )
    files += glob.glob(org_files3  )
    files += glob.glob(org_files4  ) 
    files += glob.glob(org_files5  )     

    # print("COPY source to output/source dir ", files)
    tgt_files = os.path.join( source_dir, '.' )
    for i, file in enumerate(files):
        shutil.copy(file, tgt_files)
class LossSaver(object):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    def reset(self):
        self.train_losses = []
        self.val_losses = []
    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    def return_list(self):
        return self.train_losses, self.val_losses
    def save_as_csv(self, csv_file):
        df = pd.DataFrame({'Train Losses': self.train_losses, 'Validation Losses': self.val_losses})
        df.index = [f"{i+1} Epoch" for i in df.index]
        df.to_csv(csv_file, index=True)
class AverageMeter (object):
    def __init__(self):
        self.reset ()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        
def collect_image_paths(base_dataset_dir, sub_dirs=['training', 'validation', 'test']):
    train_image_path_list = []
    train_target_path_list = []
    validation_image_path_list = []
    validation_target_path_list = []
    test_image_path_list = []
    test_target_path_list = []
    
    # 서브 디렉토리 이름과 리스트 매핑
    dir_to_lists = {
        'training': (train_image_path_list, train_target_path_list),
        'validation': (validation_image_path_list, validation_target_path_list),
        'test': (test_image_path_list, test_target_path_list)
    }
    
    for sub_dir in sub_dirs:
        # 각 서브 디렉토리의 전체 경로를 생성
        full_dir_path = os.path.join(base_dataset_dir, sub_dir)
        
        # 해당 서브 디렉토리에 대한 이미지 및 마스크 리스트 선택
        image_list, target_list = dir_to_lists.get(sub_dir, (None, None))
        
        if image_list is None or target_list is None:
            print(f"Unknown sub-directory: {sub_dir}")
            continue
        
        # 서브 디렉토리 내의 파일들을 검색
        for file_name in os.listdir(full_dir_path):
            # 이미지 파일 확장자가 .png 또는 .jpg인지 확인
            if (file_name.endswith(".png") or file_name.endswith(".jpg")) and "_mask" not in file_name:
                # 이미지 파일 경로 추가
                image_list.append(os.path.join(full_dir_path, file_name))
                
                # 대응되는 마스크 파일 경로 추가 (.png 또는 .jpg에 따른 마스크 이름 처리)
                if file_name.endswith(".png"):
                    mask_file_name = file_name.replace(".png", "_mask.png")
                elif file_name.endswith(".jpg"):
                    mask_file_name = file_name.replace(".jpg", "_mask.jpg")
                
                target_list.append(os.path.join(full_dir_path, mask_file_name))
    
    return (train_image_path_list, train_target_path_list,
            validation_image_path_list, validation_target_path_list,
            test_image_path_list, test_target_path_list)

def Do_Experiment(iteration, model_name, model, train_loader, validation_loader, test_loader, Optimizer, lr,  number_of_classes, epochs, Metrics,df,device, transform):
    start = timeit.default_timer()
    train_bool=True
    test_bool=True
    if loss_function == 'Tversky Focal Loss':
        criterion=TverskyLoss()
    elif loss_function == 'DiceBCELoss':
        criterion=DiceBCELoss()
    if Optimizer=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif Optimizer == 'SGD':
        momentum = 0.9
        weight_decay = 1e-4
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum ,weight_decay=weight_decay)
    elif Optimizer =='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if lr_scheduler_args['lr_scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = lr_scheduler_args['T_max'], eta_min = lr_scheduler_args['eta_min'])
    
    os.makedirs(output_dir, exist_ok = True)
    control_random_seed(seed)
    if train_bool:
        now = datetime.now()
        Train_date=now.strftime("%y%m%d_%H%M%S")
        print('Training Start Time:',Train_date)
        best=9999
        best_epoch=1
        Early_Stop=0
        loss_saver = LossSaver()
        train_start_time = timeit.default_timer()
        for epoch in range(1, epochs+1):
            Train_Loss = train(train_loader, epoch, 
              model, criterion, optimizer, device
              )
            lr_scheduler.step()
            outputs, targets  \
            = validate(validation_loader, 
              model, criterion, device
              )
            Val_Loss = np.round(criterion(outputs,targets).cpu().numpy(),6)            
            iou = np.round(Intersection_over_Union(outputs, targets),3)
            dice = np.round(Dice_Coefficient(outputs, targets),3)
            # f1 score 계산 추가
            _, _, f1 = Confusion_Matrix(outputs, targets)
            f1 = np.round(f1,3)
            now = datetime.now()
            date=now.strftime("%y%m%d_%H%M%S")
            print(str(epoch)+'EP('+date+'):',end=' ')
            print('T_Loss: ' + str(Train_Loss), end=' ')
            print('V_Loss: ' + str(Val_Loss), end=' ')
            print('IoU: ' + str(iou), end=' ')
            print('Dice: ' + str(dice), end=' ')
            print('F1: ' + str(f1), end=' ')  # F1 score 출력 추가
            
            loss_saver.update(Train_Loss, Val_Loss)
            loss_saver.save_as_csv(f'{output_dir}/Losses_{Experiments_Time}.csv')
            if Val_Loss<best:
                Early_Stop = 0
                torch.save(model.state_dict(), f'{output_dir}/{Train_date}_{model_name}_Iter_{iteration}.pt')
                best_epoch = epoch
                best = Val_Loss
                print('Best Epoch:',best_epoch,'Loss:',Val_Loss)
            else:
                print('')
                Early_Stop+=1
            if Early_Stop>=EARLY_STOP:
                break
        train_stop_time = timeit.default_timer()
    if test_bool:
        now = datetime.now()
        date=now.strftime("%y%m%d_%H%M%S")
        print('Test Start Time:',date)
        outputs, targets, image_paths \
            = validate(test_loader, 
              model, criterion, device,
            model_path=f'{output_dir}/{Train_date}_{model_name}_Iter_{iteration}.pt',
                       return_image_paths=True
              )        
        Loss = np.round(criterion(outputs.cpu(),targets.cpu()).numpy(),6)
        pa = np.round(Pixel_Accuracy(outputs.cpu(), targets.cpu()),3)
        iou = np.round(Intersection_over_Union(outputs.cpu(), targets.cpu()),3)
        dice = np.round(Dice_Coefficient(outputs.cpu(), targets.cpu()),3)
        recall, precision, f1 = Confusion_Matrix(outputs.cpu(), targets.cpu()) 
        recall = np.round(recall, 3); precision = np.round(precision, 3); f1 = np.round(f1, 3);
                
        now = datetime.now()
        date=now.strftime("%y%m%d_%H%M%S")
        print('Best Epoch:',best_epoch)
        print('Test('+date+'): '+'Loss: ' + str(Loss),end=' ')
        print('PA: ' + str(pa), end=' ')
        print('IoU: ' + str(iou), end=' ')
        print('Dice: ' + str(dice), end=' ')
        print('Recall: ' + str(recall), end=' ')
        print('Precision: ' + str(precision), end=' ')
        print('F1 Score: ' + str(f1), end='\n')
                            
        stop = timeit.default_timer();m, s = divmod((train_stop_time - train_start_time)/epoch, 60);h, m = divmod(m, 60);Time_per_Epoch = "%02d:%02d:%02d" % (h, m, s);
        m, s = divmod(stop - start, 60);h, m = divmod(m, 60);Time = "%02d:%02d:%02d" % (h, m, s);
        total_params = sum(p.numel() for p in model.parameters()); total_params = format(total_params , ',');
        Performances = [Experiments_Time, Train_date, iteration, model_name, best, Loss, pa, iou, dice, recall, precision, f1, total_params,Time, best_epoch, Time_per_Epoch, loss_function, lr, batch_size, epochs, FILENAME]
        new_row = pd.DataFrame([Performances], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        os.makedirs(f'{output_dir}/test_outputs', exist_ok = True)
        outputs = outputs.cpu().numpy()
        for output, image_path in zip(outputs, image_paths):
            np.save(f'{output_dir}/test_outputs/{os.path.basename(image_path)}', output)
    now = datetime.now()
    date=now.strftime("%y%m%d_%H%M%S")
    print('End',date)
    
    return df
# from models.MEGANet_Res2net import MEGANet_Res2net
# model = MEGANet_Res2net(in_channels, number_of_classes)
model_dir = 'models'
# module_names = [py.replace('.py','') for py in os.listdir(model_dir)] ; module_names = list(set(module_names)-{'.ipynb_checkpoints','__pycache__'});
module_names = ['crc_test_v72_v2_convnext']
Dataset_dir = 'splits_colondb/'
 
model_names = module_names

for  module_name in module_names:
    exec(f'from {model_dir}.{module_name} import *')

iterations = [1, 10]
# train_size=0.6

in_channels = 3
number_of_classes=1
epochs = 100 # 125
EARLY_STOP = 25  #25
batch_size = 4
devices = [0,2]

optimizer = 'AdamW'
lr = 1e-4
momentum = 0.9
weight_decay = 1e-4
optim_args = {'optimizer': optimizer, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}

lr_scheduler = 'CosineAnnealingLR'
T_max = epochs
T_0 = epochs
eta_min = 1e-6
lr_scheduler_args = {'lr_scheduler': lr_scheduler, 'T_max': T_max, 'T_0': T_0, 'eta_min': eta_min}

loss_function = 'Tversky Focal Loss'
# loss_function = 'Tversky Focal Loss'CrossEntropyLoss''DiceBCELoss
reduction = 'mean'
gamma = 2.0
weight = None
loss_function_args = {'loss_function': loss_function, 'reduction': reduction, 'gamma': gamma, 'weight': weight}

iterations = [1, 10]
now = datetime.now()
Experiments_Time=now.strftime("%y%m%d_%H%M%S")
print('Experiment Start Time:',Experiments_Time)
Metrics=['Experiment Time','Train Time', 'Iteration','Model Name', 'Val_Loss', 'Test_Loss','PA', 'IoU', 'Dice', 'Recall', 'Precision', 'F1 Score', 'Total Params','Train-Predction Time','Best Epoch','Time per Epoch', 'Loss Function', 'LR', 'Batch size', '#Epochs', 'DIR']
df = pd.DataFrame(index=None, columns=Metrics)
setting = 'etis'
output_root = f'_output/output_{setting}/'
os.makedirs(output_root, exist_ok = True)
for iteration in range(iterations[0], iterations[1]+1):
    seed = iteration
    control_random_seed(seed)
 
    # 각 iteration에 맞는 split 폴더 설정
    Dataset_dir = f'splits_Etis/split{str(iteration).zfill(2)}'
  
    
    (train_image_path_list, train_target_path_list,
     validation_image_path_list, validation_target_path_list,
     test_image_path_list, test_target_path_list) = collect_image_paths(Dataset_dir)


    train_dataset = ImagesDataset(train_image_path_list, train_target_path_list, aug=False)
    validation_dataset = ImagesDataset(validation_image_path_list, validation_target_path_list, aug=False)
    test_dataset = ImagesDataset(test_image_path_list, test_target_path_list, aug=False)
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=0, pin_memory=True, shuffle=True, drop_last=True,  # drop_last=True 추가
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, 
        num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        num_workers=0, pin_memory=True,
    )
    for model_name in model_names:
        print(f'{model_name} (Iter {iteration})')
        output_dir = output_root + f'/{model_name}_Iter_{iteration}'
        copy_sourcefile(output_dir, src_dir='src')
        control_random_seed(seed)
        model=str_to_class(model_name)(in_channels, number_of_classes)
        # from models.convsegnet_v2 import convsegnet_v2
        # model = convsegnet_v2(in_channels=in_channels, out_channels=number_of_classes)
        device = torch.device("cuda:"+str(devices[0]))
        if len(devices)>1:
            model = torch.nn.DataParallel(model, device_ids = devices ).to(device)
        else:
            model = model.to(device)
        df = Do_Experiment(seed, model_name, model, train_loader, validation_loader, test_loader,  optimizer, lr,  number_of_classes, epochs, Metrics, df, device,None)
        try:
            df.to_csv(output_root+'/'+'Seg_'+setting+"_"+model_name+"_"+Experiments_Time+'.csv', index=False, header=True, encoding="cp949")
        except:
            now = datetime.now()
            tmp_date=now.strftime("%y%m%d_%H%M%S")
            df.to_csv(output_root+'/'+'Seg_'+Experiments_Time+'_'+tmp_date+'_tmp'+'.csv', index=False, header=True, encoding="cp949")

# print('End')
# os._exit(00)
# 
# CUDA_VISIBLE_DEVICES=0,2 python 250707_ours_convnext.py
