#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
エラーズ ヒルベルト判別機 (Ehlers Hilbert Discriminator) の使用例

このサンプルではジョンエラーズ氏のヒルベルト変換理論に基づく
市場状態判別機能を実演します：
- トレンドモード vs サイクルモードの判別
- リアルタイム市場状態監視
- 位相成分とDC/AC分析の可視化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非対話的バックエンドを使用
import matplotlib.pyplot as plt

# 日本語フォントの設定（もし利用可能なら）
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from indicators.ehlers_hilbert_discriminator import EhlersHilbertDiscriminator


def generate_test_data(n_points: int = 500, add_noise: bool = True) -> pd.DataFrame:
    """
    テスト用の市場データを生成
    トレンドとサイクルが混在するデータ
    """
    np.random.seed(42)
    
    # 時間軸
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
    
    # 基本トレンド
    trend = np.linspace(100, 150, n_points)
    
    # サイクル成分（複数の周期）
    t = np.linspace(0, 4*np.pi, n_points)
    cycle1 = 10 * np.sin(t)  # 長期サイクル
    cycle2 = 5 * np.sin(3*t)  # 中期サイクル
    cycle3 = 2 * np.sin(7*t)  # 短期サイクル
    
    # 市場体制の変化をシミュレート
    price = np.zeros(n_points)
    for i in range(n_points):
        if i < n_points // 3:
            # 最初はトレンド優位
            price[i] = trend[i] + cycle1[i] * 0.3 + cycle2[i] * 0.2
        elif i < 2 * n_points // 3:
            # 中間はサイクル優位
            price[i] = trend[i] * 0.3 + cycle1[i] + cycle2[i] + cycle3[i]
        else:
            # 最後は再びトレンド優位
            price[i] = trend[i] + cycle1[i] * 0.2 + cycle2[i] * 0.1
    
    # ノイズの追加
    if add_noise:
        noise = np.random.normal(0, 1, n_points)
        price += noise
    
    # OHLC風にデータを作成
    high = price + np.random.uniform(0.5, 2.0, n_points)
    low = price - np.random.uniform(0.5, 2.0, n_points)
    open_price = price + np.random.uniform(-1.0, 1.0, n_points)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': price
    })
    
    return df


def run_hilbert_discriminator_analysis():
    """エラーズ ヒルベルト判別機の分析実行"""
    
    print("🎯 エラーズ ヒルベルト判別機 分析開始")
    print("=" * 60)
    
    # テストデータの生成
    print("📊 テストデータを生成中...")
    data = generate_test_data(n_points=300, add_noise=True)
    
    # インジケーターの初期化
    print("🔧 ヒルベルト判別機を初期化中...")
    hilbert_discriminator = EhlersHilbertDiscriminator(
        src_type='close',
        filter_length=7,
        smoothing_factor=0.2,
        analysis_window=14,
        phase_rate_threshold=0.05,  # 調整済み
        dc_ac_ratio_threshold=1.2   # 調整済み
    )
    
    # 計算実行
    print("⚡ 市場状態分析を実行中...")
    result = hilbert_discriminator.calculate(data)
    
    # 結果の表示
    print(f"\n📈 計算完了 - データポイント数: {len(result.trend_mode)}")
    
    # 統計情報の表示
    trend_mode_pct = np.mean(result.trend_mode) * 100
    cycle_mode_pct = 100 - trend_mode_pct
    avg_trend_strength = np.nanmean(result.trend_strength)
    avg_cycle_strength = np.nanmean(result.cycle_strength)
    avg_confidence = np.nanmean(result.confidence)
    
    print(f"\n📊 市場状態統計:")
    print(f"   - トレンドモード: {trend_mode_pct:.1f}%")
    print(f"   - サイクルモード: {cycle_mode_pct:.1f}%")
    print(f"   - 平均トレンド強度: {avg_trend_strength:.3f}")
    print(f"   - 平均サイクル強度: {avg_cycle_strength:.3f}")
    print(f"   - 平均信頼度: {avg_confidence:.3f}")
    
    # 最新の市場状態
    current_state = hilbert_discriminator.get_current_market_state_description()
    print(f"\n🎯 現在の市場状態: {current_state}")
    
    # チャート表示
    print("\n📈 チャートを生成中...")
    create_comprehensive_chart(data, result, hilbert_discriminator)
    
    # メタデータの表示
    metadata = hilbert_discriminator.get_discriminator_metadata()
    print(f"\n🔍 判別機メタデータ:")
    for key, value in metadata.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.4f}")
        else:
            print(f"   - {key}: {value}")
    
    print("\n✅ 分析完了")
    return data, result, hilbert_discriminator


def create_comprehensive_chart(data: pd.DataFrame, result, discriminator):
    """包括的なチャートを作成"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 日付軸の準備
    dates = data['date'] if 'date' in data.columns else range(len(data))
    
    # 1. 価格チャートと市場状態
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(dates, data['close'], label='価格', color='black', linewidth=1)
    
    # トレンドモードとサイクルモードを色分け
    trend_mode = result.trend_mode
    for i in range(len(trend_mode)):
        if trend_mode[i] == 1:  # トレンドモード
            ax1.axvspan(dates[i] if hasattr(dates, '__getitem__') else i, 
                       dates[min(i+1, len(dates)-1)] if hasattr(dates, '__getitem__') else i+1,
                       alpha=0.2, color='red', label='トレンド' if i == 0 else "")
        else:  # サイクルモード
            ax1.axvspan(dates[i] if hasattr(dates, '__getitem__') else i,
                       dates[min(i+1, len(dates)-1)] if hasattr(dates, '__getitem__') else i+1,
                       alpha=0.2, color='blue', label='サイクル' if i == 0 else "")
    
    ax1.set_title('価格と市場状態判別', fontsize=12, fontweight='bold')
    ax1.set_ylabel('価格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. トレンド・サイクル強度
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(dates, result.trend_strength, label='トレンド強度', color='red', linewidth=2)
    ax2.plot(dates, result.cycle_strength, label='サイクル強度', color='blue', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='中立線')
    ax2.fill_between(dates, result.trend_strength, result.cycle_strength, 
                     alpha=0.2, color='purple')
    ax2.set_title('トレンド・サイクル強度', fontsize=12, fontweight='bold')
    ax2.set_ylabel('強度 (0-1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 位相成分 (In-Phase & Quadrature)
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(dates, result.in_phase, label='In-Phase (I)', color='green', alpha=0.8)
    ax3.plot(dates, result.quadrature, label='Quadrature (Q)', color='orange', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('ヒルベルト変換成分', fontsize=12, fontweight='bold')
    ax3.set_ylabel('振幅')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 位相レートと周波数
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(dates, result.phase_rate, label='位相レート', color='purple', linewidth=1.5)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(dates, result.frequency, label='正規化周波数', color='brown', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title('位相レートと周波数', fontsize=12, fontweight='bold')
    ax4.set_ylabel('位相レート', color='purple')
    ax4_twin.set_ylabel('周波数', color='brown')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. DC/AC成分分析
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(dates, result.dc_component, label='DC成分 (トレンド)', color='red', linewidth=2)
    ax5.plot(dates, result.ac_component, label='AC成分 (サイクル)', color='blue', linewidth=2)
    
    # DC/AC比率のプロット
    ax5_twin = ax5.twinx()
    dc_ac_ratio = np.where(result.ac_component > 1e-10, 
                          np.abs(result.dc_component) / result.ac_component, 1.0)
    ax5_twin.plot(dates, dc_ac_ratio, label='DC/AC比率', color='purple', 
                  linewidth=1, alpha=0.7, linestyle='--')
    ax5_twin.axhline(y=discriminator.dc_ac_ratio_threshold, color='purple', 
                     linestyle=':', alpha=0.7, label=f'閾値({discriminator.dc_ac_ratio_threshold})')
    
    ax5.set_title('DC/AC成分分析', fontsize=12, fontweight='bold')
    ax5.set_ylabel('成分値', color='black')
    ax5_twin.set_ylabel('DC/AC比率', color='purple')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. 信頼度と瞬間振幅
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(dates, result.confidence, label='判別信頼度', color='darkgreen', linewidth=2)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(dates, result.amplitude, label='瞬間振幅', color='orange', linewidth=1, alpha=0.7)
    
    ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='中信頼度')
    ax6.set_title('信頼度と瞬間振幅', fontsize=12, fontweight='bold')
    ax6.set_ylabel('信頼度', color='darkgreen')
    ax6_twin.set_ylabel('振幅', color='orange')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join('examples', 'output', 'ehlers_hilbert_discriminator_analysis.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 チャートを保存しました: {output_path}")
    
    # メモリクリア
    plt.close(fig)


def demonstrate_real_time_monitoring():
    """リアルタイム監視のデモンストレーション"""
    
    print("\n🔄 リアルタイム市場状態監視デモ")
    print("-" * 40)
    
    # テストデータの生成
    data = generate_test_data(n_points=100, add_noise=True)
    
    # インジケーターの初期化
    discriminator = EhlersHilbertDiscriminator(
        src_type='close',
        filter_length=7,
        smoothing_factor=0.15,
        analysis_window=12
    )
    
    # 段階的にデータを追加してリアルタイム分析をシミュレート
    for i in range(50, len(data), 10):
        partial_data = data.iloc[:i].copy()
        result = discriminator.calculate(partial_data)
        
        current_state = discriminator.get_current_market_state_description()
        trend_strength = discriminator.get_trend_strength()[-1] if discriminator.get_trend_strength() is not None else 0
        cycle_strength = discriminator.get_cycle_strength()[-1] if discriminator.get_cycle_strength() is not None else 0
        confidence = discriminator.get_confidence()[-1] if discriminator.get_confidence() is not None else 0
        
        print(f"時点 {i:3d}: {current_state}")
        print(f"         強度 -> T:{trend_strength:.3f}, C:{cycle_strength:.3f}, 信頼度:{confidence:.3f}")
        print()


def compare_parameters():
    """異なるパラメータでの比較分析"""
    
    print("\n⚖️  パラメータ比較分析")
    print("-" * 40)
    
    # テストデータの生成
    data = generate_test_data(n_points=200, add_noise=True)
    
    # 異なるパラメータ設定
    param_configs = [
        {'filter_length': 5, 'smoothing_factor': 0.1, 'analysis_window': 10, 'name': '高感度'},
        {'filter_length': 7, 'smoothing_factor': 0.2, 'analysis_window': 14, 'name': '標準'},
        {'filter_length': 11, 'smoothing_factor': 0.3, 'analysis_window': 20, 'name': '低感度'}
    ]
    
    results = {}
    
    for config in param_configs:
        name = config.pop('name')
        discriminator = EhlersHilbertDiscriminator(src_type='close', **config)
        result = discriminator.calculate(data)
        
        trend_mode_pct = np.mean(result.trend_mode) * 100
        avg_confidence = np.nanmean(result.confidence)
        
        results[name] = {
            'trend_mode_pct': trend_mode_pct,
            'avg_confidence': avg_confidence,
            'config': config
        }
        
        print(f"{name:8s}: トレンド{trend_mode_pct:5.1f}%, 信頼度{avg_confidence:.3f}")
    
    return results


if __name__ == "__main__":
    print("🚀 エラーズ ヒルベルト判別機 - 総合分析")
    print("=" * 60)
    
    try:
        # メイン分析の実行
        data, result, discriminator = run_hilbert_discriminator_analysis()
        
        # リアルタイム監視デモ
        demonstrate_real_time_monitoring()
        
        # パラメータ比較
        param_results = compare_parameters()
        
        print("\n🎉 全ての分析が完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() 