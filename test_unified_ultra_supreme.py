#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 **EHLERS UNIFIED DC + ULTRA SUPREME DFT 統合テスト** 🧪

EhlersUnifiedDC経由でUltraSupremeDFTCycleを使用するテスト
- 統合インターフェース動作確認
- 各種カルマンフィルター設定テスト
- 従来検出器との比較
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# テスト対象のインポート
from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC

# データ生成用
def generate_test_data(n_points: int = 500) -> pd.DataFrame:
    """テストデータ生成"""
    np.random.seed(42)
    
    # 時間軸
    t = np.arange(n_points)
    
    # 基本価格レベル
    base_price = 50.0
    
    # 既知の25期間サイクル
    main_cycle = 8.0 * np.sin(2 * np.pi * t / 25)
    
    # 副次サイクル
    secondary_cycle = 3.0 * np.sin(2 * np.pi * t / 15 + np.pi/3)
    
    # トレンド
    trend = 0.01 * t
    
    # ノイズ
    noise = np.random.normal(0, 2, n_points)
    
    # 合成価格
    close_price = base_price + trend + main_cycle + secondary_cycle + noise
    
    # OHLC生成
    high = close_price + np.abs(np.random.normal(0, 1, n_points))
    low = close_price - np.abs(np.random.normal(0, 1, n_points))
    open_price = np.roll(close_price, 1)
    open_price[0] = close_price[0]
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price,
        'volume': np.random.lognormal(8, 0.2, n_points)
    })

def test_unified_dc_detectors():
    """
    EhlersUnifiedDC経由で各種検出器をテスト
    """
    print("🔄 EhlersUnifiedDC 検出器統合テスト開始...")
    
    # テストデータ生成
    data = generate_test_data(500)
    
    # テスト対象検出器
    test_detectors = [
        'dft_dominant',           # 従来DFT
        'ultra_supreme_dft',      # 🚀 新Ultra Supreme DFT
        'hody_e',                 # 拡張ホモダイン
        'phac_e',                 # 拡張位相累積
        'practical'               # 実践的検出器
    ]
    
    results = {}
    
    for detector_name in test_detectors:
        print(f"\n  📊 {detector_name} テスト...")
        
        try:
            # 検出器初期化
            detector = EhlersUnifiedDC(
                detector_type=detector_name,
                cycle_part=0.5,
                max_output=50,
                min_output=8,
                src_type='hlc3',
                use_kalman_filter=True,
                kalman_filter_type='neural_supreme',  # 統一カルマンフィルター
                window=50  # DFT窓長
            )
            
            # 性能測定
            start_time = time.time()
            cycles = detector.calculate(data)
            execution_time = time.time() - start_time
            
            # 統計計算
            stable_cycles = cycles[100:-50]  # 安定期間
            avg_cycle = np.mean(stable_cycles)
            std_cycle = np.std(stable_cycles)
            
            # 25期間真値との誤差
            true_cycle = 25.0
            error_abs = abs(avg_cycle - true_cycle)
            error_rel = error_abs / true_cycle * 100
            
            results[detector_name] = {
                'execution_time': execution_time,
                'avg_cycle': avg_cycle,
                'std_cycle': std_cycle,
                'abs_error': error_abs,
                'rel_error_pct': error_rel,
                'cycles': cycles,
                'stability_cv': std_cycle / avg_cycle if avg_cycle > 0 else float('inf')
            }
            
            print(f"    実行時間: {execution_time:.4f}秒")
            print(f"    平均サイクル: {avg_cycle:.2f} (真値: 25.0)")
            print(f"    相対誤差: {error_rel:.1f}%")
            print(f"    安定性(CV): {std_cycle/avg_cycle:.3f}")
            
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results[detector_name] = {'error': str(e)}
    
    return results, data

def test_kalman_filter_types():
    """
    Ultra Supreme DFT + 各種カルマンフィルターのテスト
    """
    print("\n🧠 Ultra Supreme DFT + カルマンフィルター組み合わせテスト...")
    
    # テストデータ
    data = generate_test_data(400)
    
    # カルマンフィルタータイプ
    kalman_types = [
        'adaptive',
        'neural_supreme', 
        'market_adaptive_unscented',
        'quantum_adaptive',
        'unscented'
    ]
    
    results = {}
    
    for kalman_type in kalman_types:
        print(f"\n  🔬 Ultra Supreme DFT + {kalman_type} テスト...")
        
        try:
            detector = EhlersUnifiedDC(
                detector_type='ultra_supreme_dft',
                cycle_part=0.5,
                max_output=40,
                min_output=10,
                src_type='hlc3',
                use_kalman_filter=True,
                kalman_filter_type=kalman_type,
                window=45
            )
            
            start_time = time.time()
            cycles = detector.calculate(data)
            execution_time = time.time() - start_time
            
            # 統計
            stable_cycles = cycles[80:-40]
            avg_cycle = np.mean(stable_cycles)
            std_cycle = np.std(stable_cycles)
            
            results[kalman_type] = {
                'execution_time': execution_time,
                'avg_cycle': avg_cycle,
                'std_cycle': std_cycle,
                'cycles': cycles
            }
            
            print(f"    実行時間: {execution_time:.4f}秒")
            print(f"    平均サイクル: {avg_cycle:.2f}")
            print(f"    標準偏差: {std_cycle:.2f}")
            
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results[kalman_type] = {'error': str(e)}
    
    return results

def create_comparison_chart(detector_results: Dict, data: pd.DataFrame):
    """
    比較チャート作成
    """  
    print("\n📊 比較チャート作成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🚀 EhlersUnifiedDC + Ultra Supreme DFT 統合テスト結果', fontsize=16)
    
    # === 1. 時系列比較 ===
    ax1 = axes[0, 0]
    
    # 成功した検出器のみプロット
    plotted_detectors = []
    for name, result in detector_results.items():
        if 'cycles' in result and 'error' not in result:
            ax1.plot(result['cycles'], label=name, alpha=0.8)
            plotted_detectors.append(name)
    
    ax1.axhline(y=25, color='red', linestyle='--', label='True Cycle (25)', linewidth=2)
    ax1.set_title('Detected Cycles Comparison')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cycle Period')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 2. 精度比較 ===
    ax2 = axes[0, 1]
    
    valid_results = {k: v for k, v in detector_results.items() if 'rel_error_pct' in v}
    if valid_results:
        names = list(valid_results.keys())
        errors = [valid_results[name]['rel_error_pct'] for name in names]
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffa726'][:len(names)]
        bars = ax2.bar(names, errors, color=colors)
        ax2.set_title('Relative Error Comparison')
        ax2.set_ylabel('Relative Error (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # バーに値を表示
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{error:.1f}%', ha='center', va='bottom')
    
    # === 3. 実行時間比較 ===
    ax3 = axes[1, 0]
    
    if valid_results:
        names = list(valid_results.keys())
        times = [valid_results[name]['execution_time'] for name in names]
        
        bars = ax3.bar(names, times, color=colors)
        ax3.set_title('Execution Time Comparison')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
    
    # === 4. テストデータ ===
    ax4 = axes[1, 1]
    
    ax4.plot(data['close'], label='Test Price', color='black', alpha=0.7)
    ax4.set_title('Test Data (25-Period Cycle)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    chart_filename = 'unified_ultra_supreme_comparison.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"📊 チャート保存: {chart_filename}")
    
    return chart_filename

def test_available_detectors():
    """
    利用可能な検出器一覧表示
    """
    print("\n📋 利用可能な検出器一覧:")
    print("=" * 60)
    
    available = EhlersUnifiedDC.get_available_detectors()
    
    for detector_name, description in available.items():
        print(f"  • {detector_name}: {description}")
    
    print(f"\n📊 総検出器数: {len(available)}")
    
    # Ultra Supreme DFTの有無確認
    if 'ultra_supreme_dft' in available:
        print("✅ 🚀🧠 Ultra Supreme DFT が正常に統合されました！")
    else:
        print("❌ Ultra Supreme DFT の統合に問題があります")

def main():
    """
    メインテスト実行
    """
    print("🚀🧠 EHLERS UNIFIED DC + ULTRA SUPREME DFT 統合テスト")
    print("=" * 80)
    
    # === 1. 利用可能検出器確認 ===
    test_available_detectors()
    
    # === 2. 統合検出器テスト ===
    detector_results, test_data = test_unified_dc_detectors()
    
    # === 3. カルマンフィルター組み合わせテスト ===
    kalman_results = test_kalman_filter_types()
    
    # === 4. 比較チャート作成 ===
    chart_file = create_comparison_chart(detector_results, test_data)
    
    # === 5. 総合結果サマリー ===
    print("\n" + "="*80)
    print("🏆 統合テスト結果サマリー")
    print("="*80)
    
    # 精度ランキング
    valid_detectors = {k: v for k, v in detector_results.items() if 'rel_error_pct' in v}
    if valid_detectors:
        print("\n📊 精度ランキング (相対誤差ベース):")
        accuracy_ranking = sorted(valid_detectors.items(), 
                                key=lambda x: x[1]['rel_error_pct'])
        for i, (name, result) in enumerate(accuracy_ranking, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            print(f"  {emoji} {i}. {name}: {result['rel_error_pct']:.1f}% 誤差")
    
    # 速度ランキング  
    if valid_detectors:
        print("\n⚡ 実行速度ランキング:")
        speed_ranking = sorted(valid_detectors.items(),
                             key=lambda x: x[1]['execution_time'])
        for i, (name, result) in enumerate(speed_ranking, 1):
            emoji = "🚀" if i == 1 else "⚡" if i == 2 else "🏃" if i == 3 else "🐌"
            print(f"  {emoji} {i}. {name}: {result['execution_time']:.4f}秒")
    
    # カルマンフィルター性能
    valid_kalman = {k: v for k, v in kalman_results.items() if 'error' not in v}
    if valid_kalman:
        print(f"\n🧠 Ultra Supreme DFT + カルマンフィルター性能:")
        kalman_ranking = sorted(valid_kalman.items(), 
                              key=lambda x: x[1]['execution_time'])
        for i, (name, result) in enumerate(kalman_ranking, 1):
            print(f"  {i}. {name}: {result['execution_time']:.4f}秒 | 平均サイクル: {result['avg_cycle']:.2f}")
    
    print(f"\n📊 比較チャート: {chart_file}")
    print("\n✅ 統合テスト完了!")
    
    return {
        'detector_results': detector_results,
        'kalman_results': kalman_results,
        'chart_file': chart_file
    }

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ 統合テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()