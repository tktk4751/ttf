#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **UKF統合版PriceSource使用例** 🎯

無香料カルマンフィルターを価格ソースとして使用する例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.price_source import PriceSource

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """サンプル価格データを作成"""
    np.random.seed(42)
    
    # 基本価格トレンド
    base_price = 100.0
    trend = np.linspace(0, 20, n_points)
    
    # ノイズとボラティリティ
    noise = np.random.normal(0, 2.0, n_points)
    volatility = np.random.normal(0, 1.0, n_points)
    
    # 価格計算
    prices = base_price + trend + noise + volatility
    
    # OHLC データ生成
    highs = prices + np.abs(np.random.normal(0, 1, n_points))
    lows = prices - np.abs(np.random.normal(0, 1, n_points))
    opens = prices + np.random.normal(0, 0.5, n_points)
    
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices
    })

def demonstrate_ukf_sources():
    """UKF価格ソースのデモンストレーション"""
    print("🚀 UKF統合版PriceSource デモンストレーション")
    print("=" * 50)
    
    # サンプルデータ作成
    data = create_sample_data()
    print(f"📊 サンプルデータ作成: {len(data)}ポイント")
    
    # 利用可能なソースタイプを表示
    print("\n📋 利用可能な価格ソース:")
    sources = PriceSource.get_available_sources()
    for src_type, description in sources.items():
        print(f"  - {src_type}: {description}")
    
    # 各ソースタイプの計算とプロット
    plt.figure(figsize=(15, 12))
    
    # 基本ソース
    basic_sources = ['close', 'hlc3', 'hl2']
    ukf_sources = ['ukf', 'ukf_hlc3', 'ukf_hl2']
    
    all_sources = basic_sources + ukf_sources
    n_plots = len(all_sources)
    
    for i, src_type in enumerate(all_sources):
        plt.subplot((n_plots + 1) // 2, 2, i + 1)
        
        try:
            # 価格計算
            prices = PriceSource.calculate_source(data, src_type)
            
            # プロット
            plt.plot(prices, label=src_type, linewidth=1)
            plt.title(f'{src_type.upper()} 価格ソース')
            plt.xlabel('時間')
            plt.ylabel('価格')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            print(f"✅ {src_type}: 計算成功 (平均値: {np.mean(prices):.2f})")
            
        except Exception as e:
            print(f"❌ {src_type}: エラー - {str(e)}")
    
    plt.tight_layout()
    plt.savefig('output/ukf_price_sources_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 比較チャートを保存: output/ukf_price_sources_comparison.png")
    
    return data, sources

def demonstrate_ukf_parameters():
    """UKFパラメータのカスタマイズ例"""
    print("\n🔧 UKFパラメータのカスタマイズ")
    print("-" * 30)
    
    data = create_sample_data(500)
    
    # 異なるUKFパラメータでテスト
    ukf_configs = [
        {'alpha': 0.001, 'process_noise_scale': 0.001, 'name': '低ノイズ'},
        {'alpha': 0.01, 'process_noise_scale': 0.01, 'name': '中ノイズ'},
        {'alpha': 0.1, 'process_noise_scale': 0.1, 'name': '高ノイズ'}
    ]
    
    plt.figure(figsize=(12, 8))
    
    # 元の価格
    original_prices = PriceSource.calculate_source(data, 'close')
    plt.plot(original_prices, label='元の価格', alpha=0.7, color='gray')
    
    for i, config in enumerate(ukf_configs):
        ukf_params = {
            'alpha': config['alpha'],
            'process_noise_scale': config['process_noise_scale']
        }
        
        # UKF価格計算
        ukf_prices = PriceSource.calculate_source(data, 'ukf', ukf_params)
        plt.plot(ukf_prices, label=f"UKF ({config['name']})", linewidth=1.5)
        
        print(f"  📈 {config['name']}: α={config['alpha']}, ノイズ={config['process_noise_scale']}")
    
    plt.title('UKFパラメータ比較')
    plt.xlabel('時間')
    plt.ylabel('価格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/ukf_parameters_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 パラメータ比較チャートを保存: output/ukf_parameters_comparison.png")

def demonstrate_ukf_full_result():
    """UKF完全結果の取得例"""
    print("\n🔍 UKF完全結果の取得")
    print("-" * 25)
    
    data = create_sample_data(300)
    
    # UKFの完全結果を取得
    ukf_result = PriceSource.get_ukf_result(data, 'close')
    
    if ukf_result is not None:
        print("✅ UKF完全結果を取得:")
        print(f"  - フィルター済み価格: {len(ukf_result.filtered_values)}ポイント")
        print(f"  - 速度推定: 平均 {np.mean(np.abs(ukf_result.velocity_estimates)):.4f}")
        print(f"  - 加速度推定: 平均 {np.mean(np.abs(ukf_result.acceleration_estimates)):.4f}")
        print(f"  - 平均不確実性: {np.mean(ukf_result.uncertainty):.4f}")
        print(f"  - 平均信頼度: {np.mean(ukf_result.confidence_scores):.4f}")
        
        # 詳細プロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # フィルター済み vs 元の価格
        axes[0, 0].plot(ukf_result.raw_values, label='元の価格', alpha=0.7)
        axes[0, 0].plot(ukf_result.filtered_values, label='UKFフィルター済み', linewidth=2)
        axes[0, 0].set_title('価格フィルタリング')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 速度推定
        axes[0, 1].plot(ukf_result.velocity_estimates, color='green')
        axes[0, 1].set_title('速度推定')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 不確実性
        axes[1, 0].plot(ukf_result.uncertainty, color='red')
        axes[1, 0].set_title('推定不確実性')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 信頼度
        axes[1, 1].plot(ukf_result.confidence_scores, color='blue')
        axes[1, 1].set_title('信頼度スコア')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/ukf_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 詳細分析チャートを保存: output/ukf_detailed_analysis.png")
    else:
        print("❌ UKF結果の取得に失敗")

def main():
    """メイン関数"""
    # 出力ディレクトリを作成
    os.makedirs('output', exist_ok=True)
    
    try:
        # 基本デモ
        data, sources = demonstrate_ukf_sources()
        
        # パラメータデモ
        demonstrate_ukf_parameters()
        
        # 完全結果デモ
        demonstrate_ukf_full_result()
        
        print("\n🎉 すべてのデモンストレーションが完了しました！")
        
        # キャッシュクリア
        PriceSource.clear_ukf_cache()
        print("🧹 UKFキャッシュをクリアしました")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 