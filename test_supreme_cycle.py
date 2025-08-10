#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supreme Cycle Detectorのテストスクリプト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import yfinance as yf

# インジケーターのインポート
from indicators.supreme_cycle_detector import SupremeCycleDetector
from indicators.ehlers_hody_dce import EhlersHoDyDCE
from indicators.ehlers_dudi_dce import EhlersDuDiDCE
from indicators.ehlers_phac_dce import EhlersPhAcDCE
from indicators.ehlers_dft_dominant_cycle import EhlersDFTDominantCycle


def fetch_market_data(symbol='BTC-USD', period='6mo', interval='4h'):
    """市場データを取得"""
    print(f"Fetching {symbol} data...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data retrieved for {symbol}")
    
    # カラム名を小文字に変換
    data.columns = [col.lower() for col in data.columns]
    
    return data


def create_synthetic_data(n_points=500):
    """合成データを作成（複数のサイクルを含む）"""
    t = np.linspace(0, 100, n_points)
    
    # 複数のサイクル成分
    cycle1 = np.sin(2 * np.pi * t / 20)  # 20期間サイクル
    cycle2 = 0.5 * np.sin(2 * np.pi * t / 35)  # 35期間サイクル
    cycle3 = 0.3 * np.sin(2 * np.pi * t / 60)  # 60期間サイクル
    
    # トレンド成分
    trend = 0.02 * t
    
    # ノイズ
    noise = 0.2 * np.random.randn(n_points)
    
    # 合成価格
    price = 100 + trend + cycle1 + cycle2 + cycle3 + noise
    
    # OHLCデータの作成
    high = price + np.abs(np.random.randn(n_points) * 0.5)
    low = price - np.abs(np.random.randn(n_points) * 0.5)
    open_price = np.roll(price, 1)
    open_price[0] = price[0]
    
    dates = [datetime.now() - timedelta(hours=4*i) for i in range(n_points, 0, -1)]
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': np.random.randint(1000, 10000, n_points)
    }, index=dates)
    
    return df


def test_supreme_cycle():
    """Supreme Cycle Detectorのテスト"""
    
    # データの選択
    use_real_data = True
    
    if use_real_data:
        try:
            data = fetch_market_data('BTC-USD', '3mo', '1h')
            title_suffix = "BTC-USD (1H)"
        except Exception as e:
            print(f"Failed to fetch real data: {e}")
            data = create_synthetic_data(500)
            title_suffix = "Synthetic Data"
    else:
        data = create_synthetic_data(500)
        title_suffix = "Synthetic Data"
    
    print(f"Data shape: {data.shape}")
    
    # インジケーターの初期化
    print("Initializing indicators...")
    
    # Supreme Cycle Detector（異なる設定）
    supreme_default = SupremeCycleDetector(
        src_type='hlc3',
        use_ukf=False,
        adaptive_params=False
    )
    
    supreme_ukf = SupremeCycleDetector(
        src_type='us_hlc3',  # Ultimate Smoother HLC3
        use_ukf=True,
        adaptive_params=True
    )
    
    # 個別コンポーネント
    hody = EhlersHoDyDCE(src_type='hlc3')
    dudi = EhlersDuDiDCE(src_type='hlc3')
    phac = EhlersPhAcDCE(src_type='hlc3')
    dft = EhlersDFTDominantCycle(src_type='hlc3')
    
    # 計算実行
    print("Calculating cycles...")
    supreme_cycles_default = supreme_default.calculate(data)
    supreme_cycles_ukf = supreme_ukf.calculate(data)
    
    hody_cycles = hody.calculate(data)
    dudi_cycles = dudi.calculate(data)
    phac_cycles = phac.calculate(data)
    dft_cycles = dft.calculate(data)
    
    # 結果の取得
    supreme_result = supreme_ukf._result
    
    # プロット作成
    print("Creating plots...")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(5, 2, height_ratios=[2, 1.5, 1.5, 1, 1])
    
    # 1. 価格チャート
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data.index, data['close'], 'b-', linewidth=1, label='Close Price')
    ax1.set_title(f'Supreme Cycle Detector Analysis - {title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Supreme Cycleの比較
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(data.index, supreme_cycles_default, 'b-', linewidth=2, label='Supreme (Default)', alpha=0.7)
    ax2.plot(data.index, supreme_cycles_ukf, 'r-', linewidth=2, label='Supreme (UKF+Adaptive)')
    ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='20 Period')
    ax2.set_ylabel('Cycle Period')
    ax2.set_title('Supreme Cycle Detection Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 50)
    
    # 3. コンポーネントサイクル
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(data.index, hody_cycles, 'g-', linewidth=1, alpha=0.7, label='HoDy')
    ax3.plot(data.index, dudi_cycles, 'b-', linewidth=1, alpha=0.7, label='DuDi')
    ax3.plot(data.index, phac_cycles, 'm-', linewidth=1, alpha=0.7, label='PhAc')
    ax3.plot(data.index, dft_cycles, 'c-', linewidth=1, alpha=0.7, label='DFT')
    ax3.plot(data.index, supreme_cycles_ukf, 'r-', linewidth=2, label='Supreme')
    ax3.set_ylabel('Cycle Period')
    ax3.set_title('Component Cycles Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=5)
    ax3.set_ylim(0, 50)
    
    # 4. 重みの時系列変化
    ax4 = fig.add_subplot(gs[3, 0])
    if supreme_result and hasattr(supreme_result, 'weights'):
        weights = supreme_result.weights
        ax4.plot(data.index, weights['hody'], 'g-', label='HoDy')
        ax4.plot(data.index, weights['dudi'], 'b-', label='DuDi')
        ax4.plot(data.index, weights['phac'], 'm-', label='PhAc')
        ax4.plot(data.index, weights['dft'], 'c-', label='DFT')
        ax4.set_ylabel('Weight')
        ax4.set_title('Component Weights Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1)
    
    # 5. 信頼度
    ax5 = fig.add_subplot(gs[3, 1])
    if supreme_result and hasattr(supreme_result, 'confidence'):
        ax5.plot(data.index, supreme_result.confidence, 'orange', linewidth=2)
        ax5.fill_between(data.index, 0, supreme_result.confidence, alpha=0.3, color='orange')
        ax5.set_ylabel('Confidence')
        ax5.set_title('Detection Confidence')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
    
    # 6. ボラティリティ状態
    ax6 = fig.add_subplot(gs[4, :])
    if supreme_result and hasattr(supreme_result, 'volatility_state'):
        vol_state = supreme_result.volatility_state
        colors = ['green', 'yellow', 'red']
        labels = ['Low', 'Medium', 'High']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = vol_state == i
            if np.any(mask):
                ax6.scatter(data.index[mask], np.ones(np.sum(mask)) * i, 
                          c=color, s=10, label=f'{label} Volatility', alpha=0.6)
        
        ax6.set_ylabel('Volatility State')
        ax6.set_title('Market Volatility State')
        ax6.set_ylim(-0.5, 2.5)
        ax6.set_yticks([0, 1, 2])
        ax6.set_yticklabels(['Low', 'Medium', 'High'])
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    plt.tight_layout()
    
    # 統計情報の表示
    if supreme_ukf._result:
        info = supreme_ukf.get_component_info()
        if info:
            print("\n=== Supreme Cycle Detector Statistics ===")
            print(f"Average Confidence: {info['average_confidence']:.2%}")
            print("\nAverage Component Weights:")
            for comp, weight in info['component_weights'].items():
                print(f"  {comp.upper()}: {weight:.2%}")
            print("\nVolatility Distribution:")
            for state, ratio in info['volatility_distribution'].items():
                print(f"  {state.capitalize()}: {ratio:.2%}")
            
            # 最新の最適コンポーネント
            best_comp = supreme_ukf.get_best_component()
            print(f"\nCurrent Best Component: {best_comp.upper()}")
    
    # プロットを保存
    plt.savefig('supreme_cycle_detector_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'supreme_cycle_detector_analysis.png'")
    
    plt.show()
    
    # サイクル値の統計
    print("\n=== Cycle Statistics ===")
    print(f"Supreme (Default) - Mean: {np.mean(supreme_cycles_default):.1f}, Std: {np.std(supreme_cycles_default):.1f}")
    print(f"Supreme (UKF)     - Mean: {np.mean(supreme_cycles_ukf):.1f}, Std: {np.std(supreme_cycles_ukf):.1f}")
    print(f"HoDy              - Mean: {np.mean(hody_cycles):.1f}, Std: {np.std(hody_cycles):.1f}")
    print(f"DuDi              - Mean: {np.mean(dudi_cycles):.1f}, Std: {np.std(dudi_cycles):.1f}")
    print(f"PhAc              - Mean: {np.mean(phac_cycles):.1f}, Std: {np.std(phac_cycles):.1f}")
    print(f"DFT               - Mean: {np.mean(dft_cycles):.1f}, Std: {np.std(dft_cycles):.1f}")


if __name__ == "__main__":
    test_supreme_cycle()