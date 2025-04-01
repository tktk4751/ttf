#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
サイクル効率比（CycleEfficiencyRatio）の使用例

このスクリプトは、サイクル効率比（CER）インジケーターの基本的な使い方を示します。
CERは様々なエーラーズのドミナントサイクル検出アルゴリズムで検出したサイクル期間を使用して、
効率比（EfficiencyRatio）を計算するインジケーターです。

このサンプルではグラフ描画は行わず、各サイクル検出器によるCERの統計情報を出力します。
"""

import numpy as np
import pandas as pd
import sys
import os
import yaml
from pathlib import Path

# インポートパスの設定
# プロジェクトのルートディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# ルートディレクトリをパスに追加
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# データ取得用のクラスをインポート
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーターをインポート
from indicators import (
    CycleEfficiencyRatio, 
    EfficiencyRatio
)


def prepare_data():
    """設定ファイルからデータを準備する"""
    # 設定ファイルの読み込み
    config_path = Path(root_dir) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("\nデータの読み込みと処理中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初の銘柄のデータを使用
    symbol = next(iter(processed_data))
    df = processed_data[symbol]
    print(f"使用するデータ: {symbol}")
    
    # データの状態を確認
    print(f"データフレームの形状: {df.shape}")
    print(f"データフレームのカラム: {df.columns.tolist()}")
    print(f"NaNの数: {df.isna().sum().sum()}")
    
    return df, symbol


def calculate_cer_statistics(data, symbol):
    """
    各サイクル検出器によるCERの統計情報を計算して表示する
    
    Args:
        data: 処理済みの価格データ（DataFrame）
        symbol: シンボル名
    """
    print(f"\n{symbol} - サイクル効率比（CER）統計分析")
    print("="*80)
    
    # 固定長効率比を初期化
    fixed_er = EfficiencyRatio(period=10)
    
    # 各サイクル検出器とCERを初期化
    detectors = {
        'dudi_dc': {'name': '二重微分', 'cer': CycleEfficiencyRatio(cycle_detector_type='dudi_dc')},
        'hody_dc': {'name': 'ホモダイン判別機', 'cer': CycleEfficiencyRatio(cycle_detector_type='hody_dc')},
        'phac_dc': {'name': '位相累積', 'cer': CycleEfficiencyRatio(cycle_detector_type='phac_dc')},
        'dudi_dce': {'name': '拡張二重微分', 'cer': CycleEfficiencyRatio(cycle_detector_type='dudi_dce')},
        'hody_dce': {'name': '拡張ホモダイン判別機', 'cer': CycleEfficiencyRatio(cycle_detector_type='hody_dce')},
        'phac_dce': {'name': '拡張位相累積', 'cer': CycleEfficiencyRatio(cycle_detector_type='phac_dce')}
    }
    
    # 結果を格納するDataFrame
    result_df = pd.DataFrame()
    result_df['close'] = data['close']
    
    # 固定長効率比を計算
    print("\n固定長効率比を計算中...")
    result_df['ER'] = fixed_er.calculate(data)
    
    # 各サイクル検出器によるCERを計算
    print("\n各サイクル検出器によるサイクル効率比を計算中...")
    for code, detector in detectors.items():
        try:
            print(f"{detector['name']}({code})の計算を開始...")
            cer_values = detector['cer'].calculate(data)
            cycles = detector['cer'].get_cycles()
            
            result_df[f'CER_{code}'] = cer_values
            result_df[f'Cycle_{code}'] = cycles
            
            print(f"  計算完了 - 配列の長さ: {len(cer_values)}, NaN数: {np.isnan(cer_values).sum()}")
        except Exception as e:
            print(f"  エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # NaNを除外した有効なデータのみを使用
    valid_data = result_df.dropna()
    if len(valid_data) == 0:
        print("\n有効なデータがありません。統計情報を計算できません。")
        return
    
    print(f"\n有効なデータ数: {len(valid_data)} / {len(result_df)} ({len(valid_data)/len(result_df)*100:.2f}%)")
    
    # 基本統計情報を表示
    print("\n1. 基本統計情報")
    print("-"*80)
    
    # CERの統計情報
    print("\n■ CER値の基本統計量:")
    cer_cols = ['ER'] + [f'CER_{code}' for code in detectors.keys() if f'CER_{code}' in valid_data.columns]
    cer_stats = valid_data[cer_cols].describe().T
    
    # 表示用に整形
    cer_stats.index = ['ER(固定期間10)'] + [f"CER({detectors[code]['name']})" for code in detectors.keys() if f'CER_{code}' in valid_data.columns]
    print(cer_stats.round(4))
    
    # サイクル期間の統計情報
    print("\n■ サイクル期間の基本統計量:")
    cycle_cols = [f'Cycle_{code}' for code in detectors.keys() if f'Cycle_{code}' in valid_data.columns]
    cycle_stats = valid_data[cycle_cols].describe().T
    
    # 表示用に整形
    cycle_stats.index = [f"周期({detectors[code]['name']})" for code in detectors.keys() if f'Cycle_{code}' in valid_data.columns]
    print(cycle_stats.round(2))
    
    # CER値の分布情報
    print("\n2. CER値の分布")
    print("-"*80)
    
    # 分布区間の定義
    bins = [
        (0.0, 0.2, "弱いレンジ(0.0-0.2)"),
        (0.2, 0.382, "レンジ(0.2-0.382)"),
        (0.382, 0.5, "中間域(0.382-0.5)"),
        (0.5, 0.618, "中間域(0.5-0.618)"),
        (0.618, 0.8, "トレンド(0.618-0.8)"),
        (0.8, 1.0, "強いトレンド(0.8-1.0)")
    ]
    
    distribution_table = {}
    
    # 各CERの分布を計算
    for col in cer_cols:
        dist = []
        for bin_start, bin_end, bin_name in bins:
            count = ((valid_data[col] >= bin_start) & (valid_data[col] <= bin_end)).sum()
            percent = count / len(valid_data) * 100
            dist.append({'区間': bin_name, '件数': count, '割合(%)': percent})
        
        distribution_table[col] = pd.DataFrame(dist).set_index('区間')
    
    # 表示
    for col in cer_cols:
        if col == 'ER':
            display_name = 'ER(固定期間10)'
        else:
            # 列名から検出器コードを安全に抽出
            parts = col.split('_')
            if len(parts) > 1 and parts[0] == 'CER':
                detector_code = parts[1]
                if detector_code in detectors:
                    display_name = f"CER({detectors[detector_code]['name']})"
                else:
                    display_name = col  # 対応する検出器がない場合はそのまま表示
            else:
                display_name = col  # 想定外の形式の場合はそのまま表示
        
        print(f"\n■ {display_name} の分布:")
        print(distribution_table[col].round(2))
    
    # 相関分析
    print("\n3. 相関分析")
    print("-"*80)
    
    corr_matrix = valid_data[cer_cols].corr().round(4)
    
    # 表示用に列名を変更
    corr_matrix.columns = ['ER(固定期間10)'] + [f"CER({detectors[code]['name']})" for code in detectors.keys() if f'CER_{code}' in valid_data.columns]
    corr_matrix.index = corr_matrix.columns
    
    print("\n■ CER間の相関係数:")
    print(corr_matrix)
    
    # トレンド/レンジ判断の一致率
    print("\n4. トレンド/レンジ判断の一致率")
    print("-"*80)
    
    # トレンド(>0.618)およびレンジ(<0.382)の判断一致率
    trend_threshold = 0.618
    range_threshold = 0.382
    
    agreement_results = []
    
    # ERと各CERの一致率
    print("\n■ ER(固定期間10)と各CERの判断一致率:")
    for code in detectors.keys():
        cer_col = f'CER_{code}'
        if cer_col not in valid_data.columns:
            continue
            
        # トレンド判断
        er_trend = valid_data['ER'] >= trend_threshold
        cer_trend = valid_data[cer_col] >= trend_threshold
        trend_agree = (er_trend & cer_trend).sum()
        trend_disagree = (er_trend ^ cer_trend).sum()
        trend_agree_rate = trend_agree / (trend_agree + trend_disagree) * 100 if (trend_agree + trend_disagree) > 0 else 0
        
        # レンジ判断
        er_range = valid_data['ER'] <= range_threshold
        cer_range = valid_data[cer_col] <= range_threshold
        range_agree = (er_range & cer_range).sum()
        range_disagree = (er_range ^ cer_range).sum()
        range_agree_rate = range_agree / (range_agree + range_disagree) * 100 if (range_agree + range_disagree) > 0 else 0
        
        # 総合一致率
        total_agree = (
            (er_trend & cer_trend).sum() + 
            (er_range & cer_range).sum() + 
            ((~er_trend & ~er_range) & (~cer_trend & ~cer_range)).sum()
        )
        total_agree_rate = total_agree / len(valid_data) * 100
        
        agreement_results.append({
            '指標': f"CER({detectors[code]['name']})",
            'トレンド一致率(%)': trend_agree_rate,
            'レンジ一致率(%)': range_agree_rate,
            '総合一致率(%)': total_agree_rate
        })
    
    # 結果表示
    agreement_df = pd.DataFrame(agreement_results).set_index('指標')
    print(agreement_df.round(2))
    
    # 結果をCSVファイルに保存
    output_dir = Path(current_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 基本統計情報
        cer_stats.to_csv(output_dir / f'cer_stats_{symbol}.csv')
        cycle_stats.to_csv(output_dir / f'cycle_period_stats_{symbol}.csv')
        
        # 相関行列
        corr_matrix.to_csv(output_dir / f'cer_correlation_{symbol}.csv')
        
        # 一致率
        agreement_df.to_csv(output_dir / f'cer_agreement_rate_{symbol}.csv')
        
        # 各CERの分布（辞書内のDataFrameをひとつずつ保存）
        for col in cer_cols:
            if col == 'ER':
                name = 'ER'
            else:
                parts = col.split('_')
                if len(parts) > 1 and parts[0] == 'CER':
                    name = parts[1]  # 安全に検出器コードを抽出
                else:
                    name = col.replace('CER_', '')  # フォールバック処理
            
            distribution_table[col].to_csv(output_dir / f'cer_distribution_{name}_{symbol}.csv')
        
        print(f"\nCSVファイルを保存しました: {output_dir}")
    except Exception as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")


def main():
    """メイン関数"""
    try:
        # データの準備
        data, symbol = prepare_data()
        
        # CER統計情報の計算と表示
        calculate_cer_statistics(data, symbol)
        
        print("\n完了しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()