import os
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

def control_random_seed(seed, pytorch=True):
    """
    실험 재현성을 위한 랜덤 시드 설정
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass

def calculate_metrics(pred, target, smooth=1e-6):
    """
    세그멘테이션 평가 지표 계산 (IoU, Dice, Precision, Recall)
    Args:
        pred: 모델 예측 마스크
        target: 실제 마스크
        smooth: 0으로 나누는 것을 방지하기 위한 평활화 값
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Positive, False Positive, False Negative 계산
    tp = torch.sum(pred * target)
    fp = torch.sum(pred) - tp
    fn = torch.sum(target) - tp
    
    # IoU (Intersection over Union) 계산
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    
    # Dice Coefficient 계산
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # Precision 계산
    precision = (tp + smooth) / (tp + fp + smooth)
    
    # Recall 계산
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def train_model(ex_dict):
    """
    모델 학습 함수
    """
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # 모델과 옵티마이저 설정
    model = ex_dict['Model'].to(ex_dict['Device'])
    optimizer = getattr(torch.optim, ex_dict['Optimizer'])(
        model.parameters(),
        lr=ex_dict['LR'],
        weight_decay=ex_dict['Weight Decay']
    )
    # 이진 세그멘테이션을 위한 BCE 손실 함수
    criterion = nn.BCEWithLogitsLoss()
    
    # 학습 중 지표 저장을 위한 딕셔너리
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    # 에포크별 학습
    for epoch in range(ex_dict['Epochs']):
        model.train()
        epoch_loss = 0
        epoch_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        
        # 배치 단위 학습
        for batch_idx, (images, masks) in enumerate(ex_dict['train_loader']):
            images = images.to(ex_dict['Device'])
            masks = masks.to(ex_dict['Device'])
            
            # 순전파
            start_time = time.time()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 배치별 평가 지표 계산
            with torch.no_grad():
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                batch_metrics = calculate_metrics(pred_masks, masks)
                
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
            
            epoch_loss += loss.item()
        
        # 에포크 평균 지표 계산
        num_batches = len(ex_dict['train_loader'])
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # 학습 지표 저장
        metrics_history['train_loss'].append(epoch_loss)
        metrics_history['train_metrics'].append(epoch_metrics)
        
        # 검증 단계
        if ex_dict['val_loader'] is not None:
            val_metrics = validate_model(model, ex_dict['val_loader'], criterion, ex_dict['Device'])
            metrics_history['val_loss'].append(val_metrics['loss'])
            metrics_history['val_metrics'].append(val_metrics)
    
    # 최종 학습 지표 저장
    ex_dict['Training_Loss'] = metrics_history['train_loss'][-1]
    ex_dict['Training_IoU'] = metrics_history['train_metrics'][-1]['iou']
    ex_dict['Training_Dice'] = metrics_history['train_metrics'][-1]['dice']
    ex_dict['Training_Precision'] = metrics_history['train_metrics'][-1]['precision']
    ex_dict['Training_Recall'] = metrics_history['train_metrics'][-1]['recall']
    
    return ex_dict

def validate_model(model, dataloader, criterion, device):
    """
    검증 데이터셋에 대한 모델 평가
    """
    model.eval()
    val_loss = 0
    val_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            batch_metrics = calculate_metrics(pred_masks, masks)
            
            val_loss += loss.item()
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]
    
    # 평균 지표 계산
    num_batches = len(dataloader)
    val_loss /= num_batches
    for key in val_metrics:
        val_metrics[key] /= num_batches
    
    val_metrics['loss'] = val_loss
    return val_metrics

def evaluate_model(ex_dict):
    """
    테스트 데이터셋에 대한 최종 모델 평가
    """
    model = ex_dict['Model'].to(ex_dict['Device'])
    model.eval()
    
    test_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    num_batches = len(ex_dict['test_loader'])
    
    with torch.no_grad():
        for images, masks in ex_dict['test_loader']:
            images = images.to(ex_dict['Device'])
            masks = masks.to(ex_dict['Device'])
            
            outputs = model(images)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            
            batch_metrics = calculate_metrics(pred_masks, masks)
            for key in test_metrics:
                test_metrics[key] += batch_metrics[key]
    
    # 평균 지표 계산
    for key in test_metrics:
        test_metrics[key] /= num_batches
    
    # 결과 저장
    ex_dict['Test_IoU'] = test_metrics['iou']
    ex_dict['Test_Dice'] = test_metrics['dice']
    ex_dict['Test_Precision'] = test_metrics['precision']
    ex_dict['Test_Recall'] = test_metrics['recall']
    
    return ex_dict

def merge_and_update_df(ex_dict):
    """
    실험 결과를 CSV 파일로 저장
    """
    results_dict = {
        'Experiment Time': ex_dict['Experiment Time'],
        'Train Time': ex_dict['Train Time'],
        'Iteration': ex_dict['Iteration'],
        'Dataset Name': ex_dict['Dataset Name'],
        'Model Name': ex_dict['Model Name'],
        'IoU': ex_dict['Test_IoU'],
        'Dice': ex_dict['Test_Dice'],
        'Precision': ex_dict['Test_Precision'],
        'Recall': ex_dict['Test_Recall'],
        'Training_IoU': ex_dict['Training_IoU'],
        'Training_Dice': ex_dict['Training_Dice'],
        'Training_Precision': ex_dict['Training_Precision'],
        'Training_Recall': ex_dict['Training_Recall'],
        'Epochs': ex_dict['Epochs'],
        'Batch Size': ex_dict['Batch Size'],
        'Device': str(ex_dict['Device']),
        'Optimizer': ex_dict['Optimizer'],
        'LR': ex_dict['LR'],
        'Weight Decay': ex_dict['Weight Decay'],
        'Momentum': ex_dict['Momentum'],
        'Image Size': ex_dict['Image Size'],
        'Output Dir': ex_dict['Output Dir'],
        'Train-Test Time': ex_dict['Train-Test Time']
    }
    
    output_csv = f"{ex_dict['Experiment Time']}_{project_name}_Results.csv"
    
    # CSV 파일 생성 또는 업데이트
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        df = pd.concat([df, pd.DataFrame([results_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([results_dict])
    
    df.to_csv(output_csv, index=False)
    return df