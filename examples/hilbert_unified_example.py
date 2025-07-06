#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌀 **Hilbert Transform Unified 使用例** 🌀

各インジケーターで使用されているヒルベルト変換アルゴリズムを
統合したクラスの使用例を示します。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.hilbert_unified import HilbertTransformUnified

# サンプルデータ生成
def generate_sample_data(n=1000):
    """サンプル価格データを生成"""
    np.random.seed(42)
    
    # 基本トレンド + ノイズ + サイクル成分
    t = np.arange(n)
    trend = 100 + 0.01 * t
    cycle = 5 * np.sin(2 * np.pi * t / 50) + 3 * np.cos(2 * np.pi * t / 30)
    noise = np.random.normal(0, 1, n)
    
    prices = trend + cycle + noise
    
    # DataFrame形式で返す
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return data

def demonstrate_algorithms():
    """各アルゴリズムの性能を比較デモンストレーション"""
    
    print("🌀 ヒルベルト変換統合システム デモンストレーション 🌀\n")
    
    # サンプルデータ生成
    data = generate_sample_data(500)
    prices = data['close'].values
    
    # 利用可能なアルゴリズムを表示
    algorithms = HilbertTransformUnified.get_available_algorithms()
    print("📋 利用可能なアルゴリズム:")
    for name, description in algorithms.items():
        print(f"  • {name}: {description}")
    print()
    
    # 各アルゴリズムでヒルベルト変換を実行
    results = {}
    
    for algorithm_name in algorithms.keys():
        print(f"🔄 {algorithm_name} を実行中...")
        
        try:
            # ヒルベルト変換インスタンス作成
            hilbert = HilbertTransformUnified(
                algorithm_type=algorithm_name,
                src_type='close'
            )
            
            # 計算実行
            result = hilbert.calculate(data)
            results[algorithm_name] = {
                'hilbert': hilbert,
                'result': result,
                'metadata': hilbert.get_algorithm_metadata()
            }
            
            print(f"  ✅ 成功: データ点数 = {len(result.amplitude)}")
            print(f"  📊 平均振幅 = {np.nanmean(result.amplitude):.4f}")
            print(f"  📊 平均周波数 = {np.nanmean(result.frequency):.6f}")
            
            # アルゴリズム固有の情報
            if result.trend_strength is not None:
                print(f"  📈 平均トレンド強度 = {np.nanmean(result.trend_strength):.4f}")
            if result.quantum_entanglement is not None:
                print(f"  🔬 平均量子もつれ = {np.nanmean(result.quantum_entanglement):.4f}")
            if result.quantum_coherence is not None:
                print(f"  🔬 平均量子コヒーレンス = {np.nanmean(result.quantum_coherence):.4f}")
            if result.confidence_score is not None:
                print(f"  🎯 平均信頼度 = {np.nanmean(result.confidence_score):.4f}")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
        
        print()
    
    return results, data

def create_comparison_chart(results, data):
    """比較チャートを作成"""
    
    print("📊 比較チャート作成中...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('🌀 ヒルベルト変換アルゴリズム比較', fontsize=16, fontweight='bold')
    
    prices = data['close'].values
    x = np.arange(len(prices))
    
    # 原価格データ
    axes[0, 0].plot(x, prices, 'b-', alpha=0.7, label='価格')
    axes[0, 0].set_title('原価格データ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 各アルゴリズムの結果をプロット
    algorithm_positions = [
        (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)
    ]
    
    for i, (algorithm_name, data_dict) in enumerate(results.items()):
        if i >= len(algorithm_positions):
            break
            
        row, col = algorithm_positions[i]
        result = data_dict['result']
        
        # 瞬時振幅をプロット
        valid_indices = ~np.isnan(result.amplitude)
        if np.any(valid_indices):
            axes[row, col].plot(x[valid_indices], result.amplitude[valid_indices], 
                              'r-', alpha=0.8, linewidth=1.5)
            axes[row, col].set_title(f'{algorithm_name}\n瞬時振幅')
            axes[row, col].grid(True, alpha=0.3)
            
            # Y軸の範囲を設定
            y_min, y_max = np.nanpercentile(result.amplitude[valid_indices], [5, 95])
            axes[row, col].set_ylim(y_min, y_max)
    
    # 最後のサブプロットで周波数比較
    if len(results) > 0:
        axes[2, 2].set_title('瞬時周波数比較')
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, (algorithm_name, data_dict) in enumerate(results.items()):
            result = data_dict['result']
            valid_indices = ~np.isnan(result.frequency)
            
            if np.any(valid_indices) and i < len(colors):
                # 周波数をスムージング
                freq_smooth = pd.Series(result.frequency[valid_indices]).rolling(window=20, center=True).mean()
                axes[2, 2].plot(x[valid_indices], freq_smooth, 
                              color=colors[i], alpha=0.7, linewidth=1.5, 
                              label=algorithm_name[:10])
        
        axes[2, 2].legend(fontsize=8)
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_ylabel('周波数')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(os.path.dirname(__file__), 'output', 'hilbert_unified_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 チャートを保存しました: {output_path}")
    
    plt.show()

def demonstrate_usage_patterns():
    """実用的な使用パターンのデモンストレーション"""
    
    print("\n🎯 実用的な使用パターン\n")
    
    # データ準備
    data = generate_sample_data(200)
    
    # パターン1: 基本的な使用
    print("1️⃣ 基本的な使用パターン:")
    hilbert_basic = HilbertTransformUnified(algorithm_type='basic')
    result_basic = hilbert_basic.calculate(data)
    
    print(f"   振幅範囲: {np.nanmin(result_basic.amplitude):.3f} - {np.nanmax(result_basic.amplitude):.3f}")
    print(f"   位相範囲: {np.nanmin(result_basic.phase):.3f} - {np.nanmax(result_basic.phase):.3f}")
    
    # パターン2: 量子強化版の使用
    print("\n2️⃣ 量子強化版の使用:")
    hilbert_quantum = HilbertTransformUnified(algorithm_type='quantum_enhanced')
    result_quantum = hilbert_quantum.calculate(data)
    
    quantum_components = hilbert_quantum.get_quantum_components()
    if quantum_components:
        print(f"   量子もつれ平均: {np.nanmean(quantum_components['quantum_entanglement']):.4f}")
    
    trend_components = hilbert_quantum.get_trend_components()
    if trend_components:
        print(f"   トレンド強度平均: {np.nanmean(trend_components['trend_strength']):.4f}")
    
    # パターン3: メタデータの活用
    print("\n3️⃣ メタデータの活用:")
    metadata = hilbert_quantum.get_algorithm_metadata()
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # パターン4: 複数アルゴリズムの組み合わせ
    print("\n4️⃣ 複数アルゴリズムの組み合わせ使用:")
    algorithms_to_test = ['basic', 'quantum_enhanced', 'quantum_supreme']
    
    ensemble_amplitude = np.zeros(len(data))
    ensemble_phase = np.zeros(len(data))
    
    for algorithm_name in algorithms_to_test:
        hilbert = HilbertTransformUnified(algorithm_type=algorithm_name)
        result = hilbert.calculate(data)
        
        # アンサンブル平均
        valid_mask = ~np.isnan(result.amplitude)
        ensemble_amplitude[valid_mask] += result.amplitude[valid_mask] / len(algorithms_to_test)
        ensemble_phase[valid_mask] += result.phase[valid_mask] / len(algorithms_to_test)
    
    print(f"   アンサンブル振幅平均: {np.nanmean(ensemble_amplitude):.4f}")
    print(f"   アンサンブル位相平均: {np.nanmean(ensemble_phase):.4f}")

def main():
    """メイン実行関数"""
    
    try:
        # アルゴリズム比較デモンストレーション
        results, data = demonstrate_algorithms()
        
        # 比較チャート作成
        if results:
            create_comparison_chart(results, data)
        
        # 実用パターンのデモンストレーション
        demonstrate_usage_patterns()
        
        print("\n🎉 ヒルベルト変換統合システムのデモンストレーションが完了しました！")
        print("\n📝 利用方法:")
        print("   • 各インジケーターから HilbertTransformUnified をインポート")
        print("   • 適切なアルゴリズムタイプを選択")
        print("   • calculate() メソッドで計算実行")
        print("   • 結果から必要な成分を取得")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 