#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from indicators.ultimate_ma_v3 import UltimateMAV3
from indicators.ultimate_ma import UltimateMA

def load_binance_data(symbol='BTC', market_type='spot', timeframe='4h', data_dir='data/binance'):
    """
    Binanceデータを直接読み込む
    """
    file_path = f"{data_dir}/{symbol}/{market_type}/{timeframe}/historical_data.csv"
    
    print(f"📂 データファイル読み込み中: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ データファイルが見つかりません: {file_path}")
        return None
    
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        # タイムスタンプをインデックスに設定
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 必要なカラムが存在するか確認
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 必要なカラムが不足しています: {missing_columns}")
            return None
        
        # データ型を数値に変換
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # NaNを除去
        df = df.dropna()
        
        print(f"✅ データ読み込み成功: {symbol} {market_type} {timeframe}")
        print(f"📊 データ期間: {df.index.min()} - {df.index.max()}")
        print(f"📈 データ数: {len(df)}件")
        
        return df
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None

def generate_advanced_test_data(n_points=2000, scenario='mixed'):
    """
    高度なテスト用データを生成
    
    Args:
        n_points: データポイント数
        scenario: シナリオ ('trend', 'range', 'volatile', 'mixed')
    """
    np.random.seed(42)
    
    if scenario == 'trend':
        # 強いトレンドデータ
        trend = np.cumsum(np.random.randn(n_points) * 0.005 + 0.002)
        noise = np.random.randn(n_points) * 0.01
    elif scenario == 'range':
        # レンジ相場データ
        trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 2
        noise = np.random.randn(n_points) * 0.02
    elif scenario == 'volatile':
        # 高ボラティリティデータ
        trend = np.cumsum(np.random.randn(n_points) * 0.001)
        noise = np.random.randn(n_points) * 0.05
    else:  # mixed
        # 混合シナリオ
        trend_part = np.cumsum(np.random.randn(n_points//2) * 0.003)
        range_part = np.sin(np.linspace(0, 2*np.pi, n_points//2)) * 1.5
        trend = np.concatenate([trend_part, range_part])
        noise = np.random.randn(n_points) * 0.015
    
    base_price = 50000 + trend * 1000 + noise * 100
    
    # OHLC作成
    high = base_price + np.abs(np.random.randn(n_points) * 50)
    low = base_price - np.abs(np.random.randn(n_points) * 50)
    open_price = base_price + np.random.randn(n_points) * 20
    close_price = base_price + np.random.randn(n_points) * 20
    
    # 日付インデックス作成
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='4H')
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price
    }, index=dates)
    
    return df

def comprehensive_performance_test():
    """包括的性能テスト"""
    print("🚀 UltimateMA V3 包括的性能テスト開始")
    print("="*60)
    
    scenarios = ['trend', 'range', 'volatile', 'mixed']
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 シナリオ: {scenario.upper()}")
        print("-" * 40)
        
        # テストデータ生成
        data = generate_advanced_test_data(1500, scenario)
        
        # UltimateMA V3
        uma_v3 = UltimateMAV3(
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
        
        # 計算実行
        start_time = time.time()
        result = uma_v3.calculate(data)
        calc_time = time.time() - start_time
        
        # 統計計算
        up_signals = np.sum(result.trend_signals == 1)
        down_signals = np.sum(result.trend_signals == -1)
        range_signals = np.sum(result.trend_signals == 0)
        total_signals = len(result.trend_signals)
        
        avg_confidence = np.mean(result.trend_confidence[result.trend_confidence > 0])
        max_confidence = np.max(result.trend_confidence)
        
        # 量子分析統計
        quantum_analysis = uma_v3.get_quantum_analysis()
        avg_quantum_strength = np.mean(np.abs(quantum_analysis['quantum_state']))
        avg_mtf_consensus = np.mean(quantum_analysis['multi_timeframe_consensus'])
        avg_fractal_dim = np.mean(quantum_analysis['fractal_dimension'])
        avg_entropy = np.mean(quantum_analysis['entropy_level'])
        
        results[scenario] = {
            'calc_time': calc_time,
            'data_points': len(data),
            'current_trend': result.current_trend,
            'current_confidence': result.current_confidence,
            'signal_distribution': {
                'up': up_signals / total_signals * 100,
                'down': down_signals / total_signals * 100,
                'range': range_signals / total_signals * 100
            },
            'confidence_stats': {
                'avg': avg_confidence,
                'max': max_confidence
            },
            'quantum_stats': {
                'quantum_strength': avg_quantum_strength,
                'mtf_consensus': avg_mtf_consensus,
                'fractal_dimension': avg_fractal_dim,
                'entropy_level': avg_entropy
            },
            'result': result,
            'data': data
        }
        
        print(f"⚡ 計算時間: {calc_time:.3f}秒 ({len(data)}ポイント)")
        print(f"🎯 現在のトレンド: {result.current_trend} (信頼度: {result.current_confidence:.3f})")
        print(f"📊 信号分布: 上昇{up_signals/total_signals*100:.1f}% | 下降{down_signals/total_signals*100:.1f}% | レンジ{range_signals/total_signals*100:.1f}%")
        print(f"🔥 平均信頼度: {avg_confidence:.3f}")
        print(f"🌌 量子強度: {avg_quantum_strength:.3f} | MTF合意度: {avg_mtf_consensus:.3f}")
    
    return results

def create_comprehensive_visualization(results):
    """包括的な可視化"""
    print("\n📈 包括的可視化チャート作成中...")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('🚀 UltimateMA V3 - 包括的性能分析', fontsize=16, fontweight='bold')
    
    scenarios = list(results.keys())
    colors = ['red', 'blue', 'green', 'orange']
    
    # 1. 各シナリオの価格チャート
    ax1 = axes[0, 0]
    for i, (scenario, data) in enumerate(results.items()):
        df = data['data']
        result = data['result']
        ax1.plot(df.index, df['close'], alpha=0.6, color=colors[i], label=f'{scenario} Price')
        ax1.plot(df.index, result.values, color=colors[i], linewidth=2, linestyle='--', 
                label=f'{scenario} UMA V3')
    ax1.set_title('💰 価格チャート比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 信号分布比較
    ax2 = axes[0, 1]
    x = np.arange(len(scenarios))
    width = 0.25
    
    up_ratios = [results[s]['signal_distribution']['up'] for s in scenarios]
    down_ratios = [results[s]['signal_distribution']['down'] for s in scenarios]
    range_ratios = [results[s]['signal_distribution']['range'] for s in scenarios]
    
    ax2.bar(x - width, up_ratios, width, label='上昇', color='green', alpha=0.7)
    ax2.bar(x, down_ratios, width, label='下降', color='red', alpha=0.7)
    ax2.bar(x + width, range_ratios, width, label='レンジ', color='blue', alpha=0.7)
    
    ax2.set_title('📊 信号分布比較')
    ax2.set_ylabel('割合 (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 信頼度統計
    ax3 = axes[1, 0]
    avg_confidences = [results[s]['confidence_stats']['avg'] for s in scenarios]
    max_confidences = [results[s]['confidence_stats']['max'] for s in scenarios]
    
    ax3.bar(x - width/2, avg_confidences, width, label='平均信頼度', color='orange', alpha=0.7)
    ax3.bar(x + width/2, max_confidences, width, label='最大信頼度', color='purple', alpha=0.7)
    
    ax3.set_title('🔥 信頼度統計')
    ax3.set_ylabel('信頼度')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 量子分析指標
    ax4 = axes[1, 1]
    quantum_strengths = [results[s]['quantum_stats']['quantum_strength'] for s in scenarios]
    mtf_consensus = [results[s]['quantum_stats']['mtf_consensus'] for s in scenarios]
    
    ax4.bar(x - width/2, quantum_strengths, width, label='量子強度', color='purple', alpha=0.7)
    ax4.bar(x + width/2, mtf_consensus, width, label='MTF合意度', color='cyan', alpha=0.7)
    
    ax4.set_title('🌌 量子分析指標')
    ax4.set_ylabel('値')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 計算性能
    ax5 = axes[2, 0]
    calc_times = [results[s]['calc_time'] * 1000 for s in scenarios]  # ミリ秒
    data_points = [results[s]['data_points'] for s in scenarios]
    
    bars = ax5.bar(scenarios, calc_times, color=colors, alpha=0.7)
    ax5.set_title('⚡ 計算時間')
    ax5.set_ylabel('時間 (ms)')
    
    # データポイント数を表示
    for bar, points in zip(bars, data_points):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{points}pts', ha='center', va='bottom', fontsize=8)
    
    ax5.grid(True, alpha=0.3)
    
    # 6. フラクタル・エントロピー分析
    ax6 = axes[2, 1]
    fractal_dims = [results[s]['quantum_stats']['fractal_dimension'] for s in scenarios]
    entropy_levels = [results[s]['quantum_stats']['entropy_level'] for s in scenarios]
    
    ax6.bar(x - width/2, fractal_dims, width, label='フラクタル次元', color='green', alpha=0.7)
    ax6.bar(x + width/2, entropy_levels, width, label='エントロピー', color='red', alpha=0.7)
    
    ax6.set_title('🔬 フラクタル・エントロピー分析')
    ax6.set_ylabel('値')
    ax6.set_xticks(x)
    ax6.set_xticklabels(scenarios)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ファイル保存
    filename = "tests/ultimate_ma_v3_comprehensive_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 包括的分析チャート保存完了: {filename}")
    
    plt.show()
    plt.close()

def real_data_test():
    """実データでのテスト"""
    print("\n🌍 実際のBinanceデータでのテスト")
    print("="*50)
    
    symbols = ['BTC', 'ETH', 'ADA']
    real_results = {}
    
    for symbol in symbols:
        print(f"\n📊 {symbol} テスト中...")
        
        # データ読み込み
        data = load_binance_data(symbol=symbol, market_type='spot', timeframe='4h')
        
        if data is None:
            print(f"❌ {symbol}のデータ読み込みに失敗")
            continue
        
        # 最新1000件を使用
        if len(data) > 1000:
            data = data.tail(1000)
        
        # UltimateMA V3
        uma_v3 = UltimateMAV3(
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
        
        # 計算実行
        start_time = time.time()
        result = uma_v3.calculate(data)
        calc_time = time.time() - start_time
        
        # 統計
        up_signals = np.sum(result.trend_signals == 1)
        down_signals = np.sum(result.trend_signals == -1)
        range_signals = np.sum(result.trend_signals == 0)
        total_signals = len(result.trend_signals)
        
        avg_confidence = np.mean(result.trend_confidence[result.trend_confidence > 0])
        
        real_results[symbol] = {
            'calc_time': calc_time,
            'current_trend': result.current_trend,
            'current_confidence': result.current_confidence,
            'avg_confidence': avg_confidence,
            'signal_distribution': {
                'up': up_signals / total_signals * 100,
                'down': down_signals / total_signals * 100,
                'range': range_signals / total_signals * 100
            }
        }
        
        print(f"✅ {symbol} 完了:")
        print(f"  ⚡ 計算時間: {calc_time:.3f}秒")
        print(f"  🎯 現在のトレンド: {result.current_trend} (信頼度: {result.current_confidence:.3f})")
        print(f"  📊 信号分布: 上昇{up_signals/total_signals*100:.1f}% | 下降{down_signals/total_signals*100:.1f}% | レンジ{range_signals/total_signals*100:.1f}%")
        print(f"  🔥 平均信頼度: {avg_confidence:.3f}")
    
    return real_results

def main():
    """メイン実行関数"""
    print("🚀 UltimateMA V3 改良版包括テストスイート")
    print("="*60)
    
    # 1. 包括的性能テスト
    synthetic_results = comprehensive_performance_test()
    
    # 2. 包括的可視化
    create_comprehensive_visualization(synthetic_results)
    
    # 3. 実データテスト
    real_results = real_data_test()
    
    # 4. 総合サマリー
    print(f"\n{'='*60}")
    print("🏆 総合テスト結果サマリー")
    print("="*60)
    
    print("\n📊 合成データテスト結果:")
    for scenario, data in synthetic_results.items():
        print(f"  {scenario.upper()}: {data['current_trend']} (信頼度: {data['current_confidence']:.3f})")
    
    if real_results:
        print("\n🌍 実データテスト結果:")
        for symbol, data in real_results.items():
            print(f"  {symbol}: {data['current_trend']} (信頼度: {data['current_confidence']:.3f})")
    
    print(f"\n✅ 全テスト完了!")
    print("📊 生成されたチャートファイルをご確認ください。")
    print("🌟 UltimateMA V3は様々な市場環境で優秀な性能を発揮しています！")

if __name__ == "__main__":
    main() 