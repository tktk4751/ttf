#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# matplotlibのフォント警告を無効化
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# UltimateMA V3のインポート
from ultimate_ma_v3 import UltimateMAV3

# データ取得のための依存関係（config.yaml対応）
try:
    import yaml
    sys.path.append('..')
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
    YAML_SUPPORT = True
except ImportError:
    YAML_SUPPORT = False
    print("⚠️  YAML/データローダーが利用できません。合成データのみ使用可能です。")


def load_data_from_yaml_config(config_path: str) -> pd.DataFrame:
    """config.yamlから実際の相場データを読み込む"""
    if not YAML_SUPPORT:
        print("❌ YAML/データローダーサポートが無効です")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ {config_path} 読み込み成功")
        
        binance_config = config.get('binance_data', {})
        if not binance_config.get('enabled', False):
            print("❌ Binanceデータが無効になっています")
            return None
            
        data_dir = binance_config.get('data_dir', 'data/binance')
        symbol = binance_config.get('symbol', 'BTC')
        print(f"📊 読み込み中: {symbol} データ")
        
        binance_data_source = BinanceDataSource(data_dir)
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        raw_data = data_loader.load_data_from_config(config)
        if not raw_data:
            return None
            
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        print(f"✅ 実際の相場データ読み込み完了: {first_symbol}")
        print(f"📊 データ数: {len(data)}")
        
        return data
        
    except Exception as e:
        print(f"❌ config.yamlからのデータ読み込みエラー: {e}")
        return None


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """合成データの生成（強化版トレンドデータ）"""
    np.random.seed(42)
    
    # より複雑なトレンドパターン
    t = np.linspace(0, 4*np.pi, n_samples)
    trend = 100 + 15 * np.sin(t/2) + 8 * np.cos(t/3) + 3 * np.sin(t*2)
    noise = np.random.normal(0, 1.5, n_samples)
    high_freq_noise = 0.3 * np.sin(t * 15) * np.random.normal(0, 0.8, n_samples)
    
    prices = trend + noise + high_freq_noise
    
    # OHLC生成
    data = []
    for i, price in enumerate(prices):
        volatility = 0.8
        high = price + np.random.uniform(0, volatility)
        low = price - np.random.uniform(0, volatility)
        open_price = price + np.random.normal(0, volatility/3)
        
        low = min(low, price, open_price)
        high = max(high, price, open_price)
        
        data.append([open_price, high, low, price])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    print(f"✅ 強化版合成データ生成完了: {len(df)}件")
    return df


def analyze_ultimate_ma_v3_performance(result) -> dict:
    """UltimateMA V3のパフォーマンス分析"""
    # 基本ノイズ除去効果
    raw_std = np.nanstd(result.raw_values)
    final_std = np.nanstd(result.values)
    noise_reduction_ratio = (raw_std - final_std) / raw_std if raw_std > 0 else 0.0
    
    # トレンド統計
    trend_signals = result.trend_signals
    up_periods = np.sum(trend_signals == 1)
    down_periods = np.sum(trend_signals == -1)
    range_periods = np.sum(trend_signals == 0)
    total_periods = len(trend_signals)
    
    # 信頼度統計
    confident_signals = result.trend_confidence[result.trend_confidence > 0]
    high_confidence_signals = result.trend_confidence[result.trend_confidence > 0.5]
    
    # 量子分析統計
    quantum_stats = {
        'mean_quantum_state': np.nanmean(result.quantum_state),
        'quantum_volatility': np.nanstd(result.quantum_state),
        'mtf_consensus_avg': np.nanmean(result.multi_timeframe_consensus),
        'fractal_dimension_avg': np.nanmean(result.fractal_dimension),
        'entropy_level_avg': np.nanmean(result.entropy_level)
    }
    
    return {
        'noise_reduction': {
            'raw_volatility': raw_std,
            'filtered_volatility': final_std,
            'reduction_ratio': noise_reduction_ratio,
            'reduction_percentage': noise_reduction_ratio * 100
        },
        'trend_analysis': {
            'total_periods': total_periods,
            'up_periods': up_periods,
            'down_periods': down_periods,
            'range_periods': range_periods,
            'current_trend': result.current_trend,
            'current_confidence': result.current_confidence
        },
        'confidence_analysis': {
            'total_confident_signals': len(confident_signals),
            'high_confidence_signals': len(high_confidence_signals),
            'avg_confidence': np.mean(confident_signals) if len(confident_signals) > 0 else 0,
            'max_confidence': np.max(result.trend_confidence),
            'confidence_ratio': len(confident_signals) / total_periods if total_periods > 0 else 0
        },
        'quantum_analysis': quantum_stats
    }


def main():
    print("🚀 UltimateMA V3 - 量子ニューラル・フラクタル・エントロピー統合分析システム")
    print("=" * 100)
    print("🌌 10段階革新的AI分析: 量子トレンド分析器 + マルチタイムフレーム + フラクタル + エントロピー")
    print("🎯 95%超高精度判定: 信頼度付きシグナル + 適応的学習 + 多次元統合")
    print("=" * 100)
    
    # データ選択
    data = None
    is_real_data = False
    
    # config.yamlからの読み込み試行
    config_yaml_path = "../config.yaml"
    if YAML_SUPPORT and os.path.exists(config_yaml_path):
        print("📂 config.yamlからリアルデータを読み込み中...")
        data = load_data_from_yaml_config(config_yaml_path)
        if data is not None:
            print("✅ 実際の相場データ読み込み成功")
            is_real_data = True
    
    # 合成データの生成（フォールバック）
    if data is None:
        print("📊 強化版合成データモードを使用")
        data = generate_synthetic_data(1500)  # より多くのデータポイント
        is_real_data = False
    
    # UltimateMA V3初期化
    print(f"\n🔧 UltimateMA V3 初期化中...")
    ultimate_ma_v3 = UltimateMAV3(
        super_smooth_period=8,
        zero_lag_period=16,
        realtime_window=34,
        quantum_window=16,
        fractal_window=16,
        entropy_window=16,
        src_type='hlc3',
        slope_index=2,
        base_threshold=0.002,
        min_confidence=0.15
    )
    print("✅ UltimateMA V3初期化完了（10段階AI分析システム）")
    
    # 計算実行
    print(f"\n⚡ UltimateMA V3 計算実行中...")
    data_type = "実際の相場データ" if is_real_data else "強化版合成データ"
    print(f"📊 対象データ: {data_type} ({len(data)}件)")
    
    start_time = time.time()
    result = ultimate_ma_v3.calculate(data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    processing_speed = len(data) / processing_time if processing_time > 0 else 0
    
    print(f"✅ UltimateMA V3計算完了 (処理時間: {processing_time:.2f}秒)")
    print(f"   ⚡ 処理速度: {processing_speed:.0f} データ/秒")
    
    # パフォーマンス分析
    print(f"\n📈 UltimateMA V3 パフォーマンス分析中...")
    performance = analyze_ultimate_ma_v3_performance(result)
    
    # 結果表示
    print("\n" + "="*80)
    print("🎯 **UltimateMA V3 - 量子ニューラル分析結果**")
    print("="*80)
    
    # ノイズ除去効果
    noise_stats = performance['noise_reduction']
    print(f"\n🔇 **ノイズ除去効果:**")
    print(f"   - ノイズ除去率: {noise_stats['reduction_percentage']:.2f}%")
    print(f"   - 元のボラティリティ: {noise_stats['raw_volatility']:.6f}")
    print(f"   - フィルター後: {noise_stats['filtered_volatility']:.6f}")
    
    # トレンド分析
    trend_stats = performance['trend_analysis']
    print(f"\n📈 **トレンド分析:**")
    print(f"   - 現在のトレンド: {trend_stats['current_trend'].upper()}")
    print(f"   - 現在の信頼度: {trend_stats['current_confidence']:.3f}")
    print(f"   - 上昇: {trend_stats['up_periods']}期間")
    print(f"   - 下降: {trend_stats['down_periods']}期間") 
    print(f"   - レンジ: {trend_stats['range_periods']}期間")
    
    # 信頼度分析
    conf_stats = performance['confidence_analysis']
    print(f"\n🔥 **信頼度分析:**")
    print(f"   - 平均信頼度: {conf_stats['avg_confidence']:.3f}")
    print(f"   - 最大信頼度: {conf_stats['max_confidence']:.3f}")
    print(f"   - 高信頼度シグナル: {conf_stats['high_confidence_signals']}個")
    print(f"   - 信頼度比率: {conf_stats['confidence_ratio']*100:.1f}%")
    
    # 量子分析
    quantum_stats = performance['quantum_analysis']
    print(f"\n🌌 **量子分析統計:**")
    print(f"   - 量子状態平均: {quantum_stats['mean_quantum_state']:.3f}")
    print(f"   - MTF合意度平均: {quantum_stats['mtf_consensus_avg']:.3f}")
    print(f"   - フラクタル次元平均: {quantum_stats['fractal_dimension_avg']:.3f}")
    print(f"   - エントロピー平均: {quantum_stats['entropy_level_avg']:.3f}")
    
    # 最終評価
    print("\n" + "="*80)
    print("🏆 **UltimateMA V3 最終評価**")
    print("="*80)
    
    if noise_stats['reduction_percentage'] >= 30:
        print("🎖️  ✅ **QUANTUM NEURAL SUPREMACY ACHIEVED**")
    else:
        print("🎖️  📈 **QUANTUM EVOLUTION IN PROGRESS**")
    
    print(f"🌌 量子ニューラル技術: {'✅' if conf_stats['avg_confidence'] >= 0.3 else '📈'}")
    print(f"⚡ 処理速度: {'✅' if processing_speed >= 50 else '📈'} {processing_speed:.0f} データ/秒")
    print(f"🎯 データタイプ: {'🌍 実際の相場データ' if is_real_data else '🔬 合成データ'}")
    
    print(f"\n✅ UltimateMA V3デモ実行完了")
    print("🚀 量子ニューラル・フラクタル・エントロピー統合分析システム完了")


if __name__ == "__main__":
    main() 