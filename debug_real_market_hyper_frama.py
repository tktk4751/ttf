#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_frama.strategy import HyperFRAMAEnhancedStrategy
from strategies.implementations.hyper_frama.signal_generator import FilterType
from data.binance_data_source import BinanceDataSource
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
import yaml

def load_real_market_data():
    """実際の市場データを読み込む（t.pyと同じ方法）"""
    print("=== 実際の市場データ読み込み ===")
    
    # 設定ファイル読み込み
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # データの準備（t.pyと同じ方法）
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
    print("  データ読み込み中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初のシンボルのデータを取得
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"✓ データ読み込み完了: {len(data)}行")
    print(f"  銘柄: {first_symbol}")
    print(f"  期間: {data.index[0]} ～ {data.index[-1]}")
    print(f"  価格範囲: ${data['close'].min():.2f} ～ ${data['close'].max():.2f}")
    
    return data

def analyze_frama_crossovers(data, strategy):
    """FRAMA値とクロスオーバーの詳細分析"""
    print("\n=== FRAMA値とクロスオーバー分析 ===")
    
    # FRAMA値とAdjusted FRAMA値を取得
    frama_values = strategy.get_frama_values(data)
    adjusted_frama_values = strategy.get_adjusted_frama_values(data)
    
    # シグナルを取得
    hyper_frama_signals = strategy.get_hyper_frama_signals(data)
    long_signals = strategy.get_long_signals(data)
    short_signals = strategy.get_short_signals(data)
    
    print(f"✓ FRAMA統計:")
    print(f"  有効データ数: {np.sum(~np.isnan(frama_values))}/{len(frama_values)}")
    print(f"  FRAMA範囲: ${np.nanmin(frama_values):.2f} ～ ${np.nanmax(frama_values):.2f}")
    print(f"  Adjusted FRAMA範囲: ${np.nanmin(adjusted_frama_values):.2f} ～ ${np.nanmax(adjusted_frama_values):.2f}")
    
    # 位置関係の分析
    valid_mask = ~(np.isnan(frama_values) | np.isnan(adjusted_frama_values))
    if np.any(valid_mask):
        frama_above_adj = frama_values[valid_mask] > adjusted_frama_values[valid_mask]
        frama_equal_adj = np.abs(frama_values[valid_mask] - adjusted_frama_values[valid_mask]) < 1e-6
        
        print(f"\n✓ 位置関係統計:")
        print(f"  FRAMA > Adjusted: {np.sum(frama_above_adj)}/{np.sum(valid_mask)} ({np.sum(frama_above_adj)/np.sum(valid_mask)*100:.1f}%)")
        print(f"  FRAMA ≈ Adjusted: {np.sum(frama_equal_adj)}/{np.sum(valid_mask)} ({np.sum(frama_equal_adj)/np.sum(valid_mask)*100:.1f}%)")
        print(f"  FRAMA < Adjusted: {np.sum(~frama_above_adj & ~frama_equal_adj)}/{np.sum(valid_mask)} ({np.sum(~frama_above_adj & ~frama_equal_adj)/np.sum(valid_mask)*100:.1f}%)")
    
    # 差の統計
    diff = frama_values - adjusted_frama_values
    valid_diff = diff[~np.isnan(diff)]
    if len(valid_diff) > 0:
        print(f"\n✓ FRAMA差分統計:")
        print(f"  平均差: ${np.mean(valid_diff):.6f}")
        print(f"  標準偏差: ${np.std(valid_diff):.6f}")
        print(f"  最大差: ${np.max(valid_diff):.6f}")
        print(f"  最小差: ${np.min(valid_diff):.6f}")
    
    # クロスオーバー発生箇所の詳細分析
    long_crossover_indices = np.where(hyper_frama_signals == 1)[0]
    short_crossover_indices = np.where(hyper_frama_signals == -1)[0]
    
    print(f"\n✓ クロスオーバー統計:")
    print(f"  HyperFRAMAロングシグナル: {len(long_crossover_indices)}")
    print(f"  HyperFRAMAショートシグナル: {len(short_crossover_indices)}")
    print(f"  実際のロングエントリー: {np.sum(long_signals)}")
    print(f"  実際のショートエントリー: {np.sum(short_signals)}")
    
    # 最初の数個の詳細分析
    if len(long_crossover_indices) > 0:
        print(f"\n  ロングクロスオーバー詳細（最初の5個）:")
        for i, idx in enumerate(long_crossover_indices[:5]):
            if idx > 0:
                prev_frama = frama_values[idx-1]
                curr_frama = frama_values[idx]
                prev_adj = adjusted_frama_values[idx-1]
                curr_adj = adjusted_frama_values[idx]
                
                prev_relation = "FRAMA > Adj" if prev_frama > prev_adj else "FRAMA <= Adj"
                curr_relation = "FRAMA > Adj" if curr_frama > curr_adj else "FRAMA <= Adj"
                
                print(f"    [{idx}] {prev_relation} -> {curr_relation}")
                print(f"         前: FRAMA=${prev_frama:.2f}, Adj=${prev_adj:.2f}, 差=${prev_frama-prev_adj:.6f}")
                print(f"         現: FRAMA=${curr_frama:.2f}, Adj=${curr_adj:.2f}, 差=${curr_frama-curr_adj:.6f}")
                print(f"         日時: {data.index[idx]}")
    
    if len(short_crossover_indices) > 0:
        print(f"\n  ショートクロスオーバー詳細（最初の5個）:")
        for i, idx in enumerate(short_crossover_indices[:5]):
            if idx > 0:
                prev_frama = frama_values[idx-1]
                curr_frama = frama_values[idx]
                prev_adj = adjusted_frama_values[idx-1]
                curr_adj = adjusted_frama_values[idx]
                
                prev_relation = "FRAMA > Adj" if prev_frama > prev_adj else "FRAMA <= Adj"
                curr_relation = "FRAMA > Adj" if curr_frama > curr_adj else "FRAMA <= Adj"
                
                print(f"    [{idx}] {prev_relation} -> {curr_relation}")
                print(f"         前: FRAMA=${prev_frama:.2f}, Adj=${prev_adj:.2f}, 差=${prev_frama-prev_adj:.6f}")
                print(f"         現: FRAMA=${curr_frama:.2f}, Adj=${curr_adj:.2f}, 差=${curr_frama-curr_adj:.6f}")
                print(f"         日時: {data.index[idx]}")
    
    return len(long_crossover_indices) == 0

def test_different_parameter_settings(data):
    """異なるパラメータ設定での比較テスト"""
    print(f"\n=== 異なるパラメータ設定での比較テスト ===")
    
    # テストパラメータ設定
    test_configs = [
        {
            "name": "デフォルト設定", 
            "params": {"period": 6, "src_type": "oc2", "fc": 4, "sc": 120, "alpha_multiplier": 0.12}
        },
        {
            "name": "敏感設定", 
            "params": {"period": 4, "src_type": "hl2", "fc": 1, "sc": 80, "alpha_multiplier": 0.08}
        },
        {
            "name": "標準設定", 
            "params": {"period": 8, "src_type": "hlc3", "fc": 2, "sc": 100, "alpha_multiplier": 0.15}
        },
        {
            "name": "より敏感設定", 
            "params": {"period": 6, "src_type": "close", "fc": 1, "sc": 60, "alpha_multiplier": 0.05}
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n  {config['name']}テスト:")
        
        strategy = HyperFRAMAEnhancedStrategy(
            position_mode=False,  # クロスオーバーモード
            filter_type=FilterType.NONE,  # フィルターなし
            **config['params']
        )
        
        # シグナル生成
        long_signals = strategy.get_long_signals(data)
        short_signals = strategy.get_short_signals(data)
        hyper_frama_signals = strategy.get_hyper_frama_signals(data)
        
        # FRAMA値の統計
        frama_values = strategy.get_frama_values(data)
        adjusted_frama_values = strategy.get_adjusted_frama_values(data)
        
        # 差分統計
        diff = frama_values - adjusted_frama_values
        valid_diff = diff[~np.isnan(diff)]
        
        long_count = np.sum(long_signals)
        short_count = np.sum(short_signals)
        hyper_long_count = np.sum(hyper_frama_signals == 1)
        hyper_short_count = np.sum(hyper_frama_signals == -1)
        
        print(f"    パラメータ: {config['params']}")
        print(f"    HyperFRAMAシグナル: ロング={hyper_long_count}, ショート={hyper_short_count}")
        print(f"    エントリーシグナル: ロング={long_count}, ショート={short_count}")
        if len(valid_diff) > 0:
            print(f"    FRAMA差分統計: 平均=${np.mean(valid_diff):.8f}, 標準偏差=${np.std(valid_diff):.8f}")
        
        results.append({
            'config': config['name'],
            'long_signals': long_count,
            'short_signals': short_count,
            'hyper_long': hyper_long_count,
            'hyper_short': hyper_short_count
        })
    
    print(f"\n  テスト結果まとめ:")
    for result in results:
        print(f"    {result['config']}: ロング={result['long_signals']}, ショート={result['short_signals']}")
    
    return any(r['long_signals'] > 0 for r in results)

def analyze_position_relationships(data, strategy):
    """位置関係の時系列分析"""
    print(f"\n=== 位置関係の時系列分析 ===")
    
    frama_values = strategy.get_frama_values(data)
    adjusted_frama_values = strategy.get_adjusted_frama_values(data)
    
    # 位置関係の変化を追跡
    position_changes = []
    current_position = None
    
    for i in range(len(frama_values)):
        if not (np.isnan(frama_values[i]) or np.isnan(adjusted_frama_values[i])):
            if frama_values[i] > adjusted_frama_values[i]:
                new_position = 1  # FRAMA > Adjusted
            else:
                new_position = -1  # FRAMA <= Adjusted
            
            if current_position is not None and current_position != new_position:
                position_changes.append({
                    'index': i,
                    'datetime': data.index[i],
                    'from': current_position,
                    'to': new_position,
                    'frama': frama_values[i],
                    'adjusted': adjusted_frama_values[i],
                    'close': data['close'].iloc[i]
                })
            
            current_position = new_position
    
    print(f"✓ 位置関係変化: {len(position_changes)}回")
    
    # ゴールデンクロス（-1 -> 1）とデッドクロス（1 -> -1）の分析
    golden_crosses = [pc for pc in position_changes if pc['from'] == -1 and pc['to'] == 1]
    dead_crosses = [pc for pc in position_changes if pc['from'] == 1 and pc['to'] == -1]
    
    print(f"  ゴールデンクロス（FRAMA上抜け）: {len(golden_crosses)}回")
    print(f"  デッドクロス（FRAMA下抜け）: {len(dead_crosses)}回")
    
    if len(golden_crosses) > 0:
        print(f"\n  ゴールデンクロス詳細（最初の10個）:")
        for i, gc in enumerate(golden_crosses[:10]):
            print(f"    [{gc['index']}] {gc['datetime']}")
            print(f"         FRAMA: ${gc['frama']:.2f} > Adjusted: ${gc['adjusted']:.2f}")
            print(f"         Close: ${gc['close']:.2f}")
    
    if len(dead_crosses) > 0:
        print(f"\n  デッドクロス詳細（最初の10個）:")
        for i, dc in enumerate(dead_crosses[:10]):
            print(f"    [{dc['index']}] {dc['datetime']}")
            print(f"         FRAMA: ${dc['frama']:.2f} < Adjusted: ${dc['adjusted']:.2f}")
            print(f"         Close: ${dc['close']:.2f}")
    
    return len(golden_crosses), len(dead_crosses)

def main():
    """メイン分析関数"""
    print("実際の市場データでのHyperFRAMA問題診断")
    print("=" * 60)
    
    try:
        # データ読み込み
        data = load_real_market_data()
        
        # デフォルト設定でのストラテジー（t.pyと同じ設定）
        strategy = HyperFRAMAEnhancedStrategy(
            position_mode=False,  # クロスオーバーモード
            filter_type=FilterType.NONE  # フィルターなし
        )
        
        print(f"\n✓ ストラテジー初期化: {strategy.name}")
        print(f"  Position Mode: {strategy._parameters['position_mode']}")
        print(f"  Filter Type: {strategy._parameters['filter_type']}")
        print(f"  Period: {strategy._parameters['period']}")
        print(f"  Source Type: {strategy._parameters['src_type']}")
        print(f"  Alpha Multiplier: {strategy._parameters['alpha_multiplier']}")
        
        # 詳細分析の実行
        no_long_crossovers = analyze_frama_crossovers(data, strategy)
        golden_crosses, dead_crosses = analyze_position_relationships(data, strategy)
        any_long_found = test_different_parameter_settings(data)
        
        # 結果まとめ
        print(f"\n{'='*60}")
        print(f"診断結果まとめ:")
        print(f"  データ期間: {len(data)}本のローソク足")
        print(f"  ゴールデンクロス: {golden_crosses}回")
        print(f"  デッドクロス: {dead_crosses}回")
        print(f"  デフォルト設定でのロング検出: {'なし' if no_long_crossovers else 'あり'}")
        print(f"  その他設定でのロング検出: {'あり' if any_long_found else 'なし'}")
        
        if no_long_crossovers and golden_crosses > 0:
            print(f"\n⚠️ 問題特定:")
            print(f"  位置関係上ではゴールデンクロスが{golden_crosses}回発生していますが、")
            print(f"  HyperFRAMAクロスオーバー検出アルゴリズムが正しく検出していません。")
            print(f"  これは位置関係ベースのクロスオーバー検出ロジックに問題がある可能性があります。")
        elif no_long_crossovers and golden_crosses == 0:
            print(f"\n✓ 分析結果:")
            print(f"  実際の市場データではFRAMAがAdjusted FRAMAを上抜けるゴールデンクロスが")
            print(f"  発生していないため、ロングシグナルが生成されないのは正常です。")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()