import pandas as pd
import numpy as np

def analyze_size_performance_relationship(excel_path):
    # 엑셀 파일의 모든 시트 읽기
    xl = pd.ExcelFile(excel_path)
    
    # 결과를 저장할 딕셔너리
    results = {}
    
    # 전체 데이터를 저장할 리스트
    all_data = []
    
    # 각 시트(데이터셋)별로 분석
    for sheet_name in xl.sheet_names:
        # 시트 데이터 읽기
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 데이터셋 이름 컬럼 추가
        df['dataset'] = sheet_name
        all_data.append(df)
        
        # polyp_size의 중앙값을 기준으로 크기 구분
        size_threshold = df['polyp_size'].median()
        
        # DICE 성능 기준 (예: 0.6 미만을 낮은 성능으로 정의)
        dice_threshold = 0.6
        
        # 작은 폴립 & 낮은 성능
        small_poor = df[(df['polyp_size'] < size_threshold) & (df['dice'] < dice_threshold)]
        
        # 큰 폴립 & 낮은 성능
        large_poor = df[(df['polyp_size'] >= size_threshold) & (df['dice'] < dice_threshold)]
        
        # 전체 케이스 수
        total_cases = len(df)
        
        results[sheet_name] = {
            'total_cases': total_cases,
            'size_threshold': size_threshold,
            'small_polyps': {
                'count': len(small_poor),
                'percentage': (len(small_poor) / total_cases) * 100,
                'mean_dice': small_poor['dice'].mean() if not small_poor.empty else 0,
                'mean_size': small_poor['polyp_size'].mean() if not small_poor.empty else 0,
                'case_ids': small_poor['case_id'].tolist()
            },
            'large_polyps': {
                'count': len(large_poor),
                'percentage': (len(large_poor) / total_cases) * 100,
                'mean_dice': large_poor['dice'].mean() if not large_poor.empty else 0,
                'mean_size': large_poor['polyp_size'].mean() if not large_poor.empty else 0,
                'case_ids': large_poor['case_id'].tolist()
            }
        }
    
    # 전체 데이터 통합
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 전체 데이터에 대한 분석
    total_size_threshold = combined_df['polyp_size'].median()
    total_small_poor = combined_df[(combined_df['polyp_size'] < total_size_threshold) & (combined_df['dice'] < dice_threshold)]
    total_large_poor = combined_df[(combined_df['polyp_size'] >= total_size_threshold) & (combined_df['dice'] < dice_threshold)]
    
    results['Combined_Analysis'] = {
        'total_cases': len(combined_df),
        'size_threshold': total_size_threshold,
        'small_polyps': {
            'count': len(total_small_poor),
            'percentage': (len(total_small_poor) / len(combined_df)) * 100,
            'mean_dice': total_small_poor['dice'].mean() if not total_small_poor.empty else 0,
            'mean_size': total_small_poor['polyp_size'].mean() if not total_small_poor.empty else 0,
            'case_details': total_small_poor.groupby('dataset')['case_id'].apply(list).to_dict()
        },
        'large_polyps': {
            'count': len(total_large_poor),
            'percentage': (len(total_large_poor) / len(combined_df)) * 100,
            'mean_dice': total_large_poor['dice'].mean() if not total_large_poor.empty else 0,
            'mean_size': total_large_poor['polyp_size'].mean() if not total_large_poor.empty else 0,
            'case_details': total_large_poor.groupby('dataset')['case_id'].apply(list).to_dict()
        }
    }
    
    # 결과를 새로운 엑셀 파일로 저장
    with pd.ExcelWriter('size_performance_analysis.xlsx') as writer:
        # 요약 시트 생성
        summary_data = []
        for dataset, data in results.items():
            if dataset != 'Combined_Analysis':
                summary_data.append({
                    'Dataset': dataset,
                    'Total Cases': data['total_cases'],
                    'Size Threshold': data['size_threshold'],
                    'Small Poor Count': data['small_polyps']['count'],
                    'Small Poor %': f"{data['small_polyps']['percentage']:.2f}%",
                    'Small Poor Mean Dice': f"{data['small_polyps']['mean_dice']:.3f}",
                    'Small Poor Mean Size': f"{data['small_polyps']['mean_size']:.1f}",
                    'Large Poor Count': data['large_polyps']['count'],
                    'Large Poor %': f"{data['large_polyps']['percentage']:.2f}%",
                    'Large Poor Mean Dice': f"{data['large_polyps']['mean_dice']:.3f}",
                    'Large Poor Mean Size': f"{data['large_polyps']['mean_size']:.1f}"
                })
        
        # 전체 분석 결과 추가
        combined_data = results['Combined_Analysis']
        summary_data.append({
            'Dataset': 'ALL DATASETS',
            'Total Cases': combined_data['total_cases'],
            'Size Threshold': combined_data['size_threshold'],
            'Small Poor Count': combined_data['small_polyps']['count'],
            'Small Poor %': f"{combined_data['small_polyps']['percentage']:.2f}%",
            'Small Poor Mean Dice': f"{combined_data['small_polyps']['mean_dice']:.3f}",
            'Small Poor Mean Size': f"{combined_data['small_polyps']['mean_size']:.1f}",
            'Large Poor Count': combined_data['large_polyps']['count'],
            'Large Poor %': f"{combined_data['large_polyps']['percentage']:.2f}%",
            'Large Poor Mean Dice': f"{combined_data['large_polyps']['mean_dice']:.3f}",
            'Large Poor Mean Size': f"{combined_data['large_polyps']['mean_size']:.1f}"
        })
        
        # 요약 데이터프레임 생성 및 저장
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 전체 데이터 상세 분석 시트
        combined_detail = pd.DataFrame({
            'Category': ['Small Polyps', 'Large Polyps'],
            'Count': [
                combined_data['small_polyps']['count'],
                combined_data['large_polyps']['count']
            ],
            'Percentage': [
                f"{combined_data['small_polyps']['percentage']:.2f}%",
                f"{combined_data['large_polyps']['percentage']:.2f}%"
            ],
            'Mean Dice': [
                f"{combined_data['small_polyps']['mean_dice']:.3f}",
                f"{combined_data['large_polyps']['mean_dice']:.3f}"
            ],
            'Mean Size': [
                f"{combined_data['small_polyps']['mean_size']:.1f}",
                f"{combined_data['large_polyps']['mean_size']:.1f}"
            ]
        })
        combined_detail.to_excel(writer, sheet_name='Combined_Analysis', index=False)
    
    # 전체 분석 결과 출력
    print("\n=== 전체 데이터셋 통합 분석 결과 ===")
    print(f"전체 케이스: {combined_data['total_cases']}")
    print(f"폴립 크기 기준값: {combined_data['size_threshold']:.1f}")
    print("\n작은 폴립 & 낮은 성능:")
    print(f"- 개수: {combined_data['small_polyps']['count']}")
    print(f"- 비율: {combined_data['small_polyps']['percentage']:.2f}%")
    print(f"- 평균 Dice: {combined_data['small_polyps']['mean_dice']:.3f}")
    print("\n큰 폴립 & 낮은 성능:")
    print(f"- 개수: {combined_data['large_polyps']['count']}")
    print(f"- 비율: {combined_data['large_polyps']['percentage']:.2f}%")
    print(f"- 평균 Dice: {combined_data['large_polyps']['mean_dice']:.3f}")
    
    return results

# 사용 예시
excel_path = '/userHome/userhome2/donghee/modelcombination/빈도.xlsx'
results = analyze_size_performance_relationship(excel_path)