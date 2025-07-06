#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum Efficiency Ratio (QER) 使用例とデモンストレーション

このスクリプトは、QERインジケーターの革新的機能を実演し、
従来のEfficiency Ratioとの比較を行います。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from indicators.quantum_efficiency_ratio import QuantumEfficiencyRatio
    from indicators.efficiency_ratio import EfficiencyRatio
    from data.data_loader import DataLoader
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("indicators/ および data/ ディレクトリが適切に配置されていることを確認してください。")
    sys.exit(1)


def generate_synthetic_data(n_points: int = 1000) -> pd.DataFrame:
    """
    テスト用の合成データを生成
    - トレンド期間
    - レンジ期間
    - ブレイクアウト期間
    - ノイズを含む複合的な価格変動
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
    base_price = 50000
    
    # 複合的な価格パターンを生成
    prices = []
    current_price = base_price
    
    for i in range(n_points):
        # サイクル成分（長期トレンド）
        trend_cycle = np.sin(2 * np.pi * i / 200) * 0.02
        
        # 中期サイクル
        medium_cycle = np.sin(2 * np.pi * i / 50) * 0.01
        
        # ボラティリティクラスター
        vol_cluster = 1.0 + 0.5 * np.sin(2 * np.pi * i / 100) ** 2
        
        # 基本的な価格変動
        if i < 200:  # トレンド期間
            drift = 0.001
        elif i < 400:  # レンジ期間
            drift = 0.0002 * np.sin(2 * np.pi * i / 30)
        elif i < 600:  # ブレイクアウト期間
            drift = 0.003 if i < 550 else -0.002
        elif i < 800:  # 下降トレンド期間
            drift = -0.0015
        else:  # 複雑なレンジ期間
            drift = 0.0005 * np.sin(2 * np.pi * i / 15)
        
        # ノイズ（市場のランダム性）
        noise = np.random.normal(0, 0.003 * vol_cluster)
        
        # 価格更新
        total_change = (drift + trend_cycle + medium_cycle + noise)
        current_price *= (1 + total_change)
        prices.append(current_price)
    
    # OHLC データを生成
    data = []
    for i, price in enumerate(prices):
        # ランダムなOHLC（Close基準）
        vol = abs(np.random.normal(0, 0.005))
        high = price * (1 + vol)
        low = price * (1 - vol)
        open_price = prices[i-1] if i > 0 else price
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(100, 1000)
        })
    
    return pd.DataFrame(data)


def compare_traditional_vs_quantum_er(data: pd.DataFrame):
    """
    従来のERとQuantum ERの比較分析
    """
    print("🔬 従来のER vs Quantum ER 比較分析")
    print("=" * 60)
    
    # 従来のER計算
    traditional_er = EfficiencyRatio(
        period=14,
        src_type='hlc3',
        use_dynamic_period=False,
        smoothing_method='hma'
    )
    
    # Quantum ER計算（フル機能）
    quantum_er = QuantumEfficiencyRatio(
        base_period=14,
        src_type='hlc3',
        use_multiscale=True,
        use_predictive=True,
        use_adaptive_filter=True
    )
    
    # Quantum ER計算（基本版）
    quantum_er_basic = QuantumEfficiencyRatio(
        base_period=14,
        src_type='hlc3',
        use_multiscale=False,
        use_predictive=False,
        use_adaptive_filter=False
    )
    
    # 計算実行
    traditional_result = traditional_er.calculate(data)
    quantum_result = quantum_er.calculate(data)
    quantum_basic_result = quantum_er_basic.calculate(data)
    
    # 結果の可視化
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # 価格チャート
    axes[0].plot(data['timestamp'], data['close'], label='価格', color='black', alpha=0.7)
    axes[0].set_title('📈 価格チャート', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ER比較
    axes[1].plot(data['timestamp'], traditional_result.values, 
                label='従来のER', color='blue', alpha=0.8, linewidth=1.5)
    axes[1].plot(data['timestamp'], quantum_basic_result.values, 
                label='QER（基本版）', color='green', alpha=0.8, linewidth=1.5)
    axes[1].plot(data['timestamp'], quantum_result.values, 
                label='QER（フル機能）', color='red', alpha=0.9, linewidth=2.0)
    
    axes[1].axhline(y=0.618, color='orange', linestyle='--', alpha=0.7, label='効率閾値(0.618)')
    axes[1].axhline(y=0.382, color='purple', linestyle='--', alpha=0.7, label='非効率閾値(0.382)')
    
    axes[1].set_title('🔍 効率比(ER)の比較', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('効率比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)
    
    # QERの多次元分析
    axes[2].plot(data['timestamp'], quantum_result.multiscale_values, 
                label='マルチスケール値', color='cyan', alpha=0.8)
    axes[2].plot(data['timestamp'], quantum_result.predictive_values, 
                label='予測的成分', color='magenta', alpha=0.8)
    axes[2].plot(data['timestamp'], quantum_result.confidence_values, 
                label='信頼度', color='gold', alpha=0.8)
    
    axes[2].set_title('🚀 QER 多次元分析', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('値')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.0)
    
    # 市場レジームとトレンド信号
    # レジーム表示用の背景色
    regime_colors = {0: 'gray', 1: 'lightblue', 2: 'lightcoral'}
    regime_names = {0: 'レンジ', 1: 'トレンド', 2: 'ブレイクアウト'}
    
    # トレンド信号のプロット
    trend_signals = quantum_result.trend_signals
    up_signals = np.where(trend_signals == 1, quantum_result.values, np.nan)
    down_signals = np.where(trend_signals == -1, quantum_result.values, np.nan)
    range_signals = np.where(trend_signals == 0, quantum_result.values, np.nan)
    
    axes[3].plot(data['timestamp'], quantum_result.values, color='black', alpha=0.5, linewidth=1)
    axes[3].scatter(data['timestamp'], up_signals, color='green', s=10, alpha=0.8, label='上昇信号')
    axes[3].scatter(data['timestamp'], down_signals, color='red', s=10, alpha=0.8, label='下降信号')
    axes[3].scatter(data['timestamp'], range_signals, color='gray', s=5, alpha=0.5, label='レンジ信号')
    
    axes[3].set_title('📊 市場レジーム & トレンド信号', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('QER値')
    axes[3].set_xlabel('時間')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 日付フォーマット調整
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('examples/output/quantum_er_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計的比較
    print("\n📊 統計的パフォーマンス比較")
    print("-" * 40)
    
    # ノイズレベル（標準偏差）の比較
    traditional_noise = np.nanstd(np.diff(traditional_result.values[50:]))
    quantum_noise = np.nanstd(np.diff(quantum_result.values[50:]))
    quantum_basic_noise = np.nanstd(np.diff(quantum_basic_result.values[50:]))
    
    print(f"従来のER ノイズレベル:     {traditional_noise:.6f}")
    print(f"QER基本版 ノイズレベル:    {quantum_basic_noise:.6f}")
    print(f"QERフル版 ノイズレベル:    {quantum_noise:.6f}")
    print(f"ノイズ削減率（基本版）:    {((traditional_noise - quantum_basic_noise) / traditional_noise * 100):.1f}%")
    print(f"ノイズ削減率（フル版）:    {((traditional_noise - quantum_noise) / traditional_noise * 100):.1f}%")
    
    # 反応速度（相関分析）
    price_returns = np.diff(data['close'].values[50:])
    traditional_diff = np.diff(traditional_result.values[50:])
    quantum_diff = np.diff(quantum_result.values[50:])
    
    traditional_corr = np.corrcoef(price_returns[1:], traditional_diff[1:])[0, 1]
    quantum_corr = np.corrcoef(price_returns[1:], quantum_diff[1:])[0, 1]
    
    print(f"\n価格変動との相関:")
    print(f"従来のER:              {traditional_corr:.4f}")
    print(f"QER:                   {quantum_corr:.4f}")
    print(f"相関向上率:            {((quantum_corr - traditional_corr) / abs(traditional_corr) * 100):.1f}%")


def demonstrate_qer_features(data: pd.DataFrame):
    """
    QERの先進機能をデモンストレーション
    """
    print("\n🌟 Quantum ER 先進機能デモンストレーション")
    print("=" * 60)
    
    # 異なる設定でのQER計算
    configs = [
        {
            'name': 'デフォルト設定',
            'params': {},
            'color': 'blue'
        },
        {
            'name': '高感度設定',
            'params': {
                'base_period': 7,
                'confidence_threshold': 0.4,
                'slope_period': 2
            },
            'color': 'red'
        },
        {
            'name': '安定性重視設定',
            'params': {
                'base_period': 21,
                'confidence_threshold': 0.8,
                'slope_period': 5,
                'cascade_periods': [5, 14, 21]
            },
            'color': 'green'
        },
        {
            'name': '予測重視設定',
            'params': {
                'momentum_period': 3,
                'trend_period': 14,
                'use_predictive': True,
                'confidence_threshold': 0.5
            },
            'color': 'purple'
        }
    ]
    
    results = {}
    for config in configs:
        qer = QuantumEfficiencyRatio(**config['params'])
        results[config['name']] = {
            'result': qer.calculate(data),
            'indicator': qer,
            'color': config['color']
        }
    
    # 結果の可視化
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. 各設定での基本QER値比較
    axes[0].plot(data['timestamp'], data['close'], color='black', alpha=0.3, label='価格')
    ax0_twin = axes[0].twinx()
    for name, result_data in results.items():
        ax0_twin.plot(data['timestamp'], result_data['result'].values, 
                     label=name, color=result_data['color'], alpha=0.8)
    
    axes[0].set_title('🎯 異なる設定でのQER比較', fontweight='bold')
    axes[0].set_ylabel('価格')
    ax0_twin.set_ylabel('QER値')
    ax0_twin.legend()
    
    # 2. 信頼度分析
    for name, result_data in results.items():
        axes[1].plot(data['timestamp'], result_data['result'].confidence_values, 
                    label=f'{name} 信頼度', color=result_data['color'], alpha=0.8)
    
    axes[1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='デフォルト閾値')
    axes[1].set_title('🎖️ 信頼度分析', fontweight='bold')
    axes[1].set_ylabel('信頼度')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    # 3. 予測成分分析
    for name, result_data in results.items():
        if np.any(result_data['result'].predictive_values > 0):
            axes[2].plot(data['timestamp'], result_data['result'].predictive_values, 
                        label=f'{name} 予測成分', color=result_data['color'], alpha=0.8)
    
    axes[2].set_title('🔮 予測成分分析', fontweight='bold')
    axes[2].set_ylabel('予測値')
    axes[2].legend()
    
    # 4. マルチスケール分析
    default_result = results['デフォルト設定']['result']
    axes[3].plot(data['timestamp'], default_result.values, label='統合QER', color='blue', linewidth=2)
    axes[3].plot(data['timestamp'], default_result.multiscale_values, 
                label='マルチスケール成分', color='cyan', alpha=0.8)
    
    axes[3].set_title('🔄 マルチスケール効率性', fontweight='bold')
    axes[3].set_ylabel('効率性値')
    axes[3].legend()
    
    # 5. 市場レジーム検出
    regime_result = default_result
    regime_data = []
    for i in range(len(data)):
        if i < len(regime_result.values):
            # レジーム情報を取得するため、最新のQERインジケーターを使用
            qer_temp = QuantumEfficiencyRatio()
            temp_result = qer_temp.calculate(data.iloc[:i+1] if i > 50 else data.iloc[:51])
            regime_data.append(temp_result.current_regime if temp_result.current_regime != 'unknown' else 0)
        else:
            regime_data.append(0)
    
    # レジームの可視化（簡易版）
    axes[4].plot(data['timestamp'], regime_result.values, color='black', alpha=0.7)
    
    # ボラティリティ状態の表示
    vol_high = np.where(regime_result.volatility_state == 1, regime_result.values, np.nan)
    vol_low = np.where(regime_result.volatility_state == 0, regime_result.values, np.nan)
    
    axes[4].scatter(data['timestamp'], vol_high, color='red', s=10, alpha=0.6, label='高ボラティリティ')
    axes[4].scatter(data['timestamp'], vol_low, color='blue', s=10, alpha=0.6, label='低ボラティリティ')
    
    axes[4].set_title('📈 ボラティリティ状態検出', fontweight='bold')
    axes[4].set_ylabel('QER値')
    axes[4].legend()
    
    # 6. パフォーマンス統計
    axes[5].axis('off')
    
    # 統計テーブルの作成
    stats_text = "📊 パフォーマンス統計\n\n"
    
    for name, result_data in results.items():
        result = result_data['result']
        indicator = result_data['indicator']
        
        # 有効データの範囲を取得
        valid_mask = ~np.isnan(result.values)
        valid_values = result.values[valid_mask]
        
        if len(valid_values) > 0:
            avg_qer = np.mean(valid_values)
            avg_confidence = np.mean(result.confidence_values[valid_mask])
            trending_ratio = np.sum(result.trend_signals != 0) / len(result.trend_signals) * 100
            
            stats_text += f"{name}:\n"
            stats_text += f"  平均QER値: {avg_qer:.3f}\n"
            stats_text += f"  平均信頼度: {avg_confidence:.3f}\n"
            stats_text += f"  トレンド検出率: {trending_ratio:.1f}%\n"
            stats_text += f"  現在のトレンド: {result.current_trend}\n"
            stats_text += f"  現在のレジーム: {result.current_regime}\n\n"
    
    axes[5].text(0.05, 0.95, stats_text, transform=axes[5].transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 日付フォーマット調整
    for i, ax in enumerate(axes[:5]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/output/quantum_er_features.png', dpi=300, bbox_inches='tight')
    plt.show()


def realtime_trading_simulation(data: pd.DataFrame):
    """
    リアルタイム取引シミュレーション
    QERを使った簡単な取引戦略のデモ
    """
    print("\n💰 リアルタイム取引シミュレーション")
    print("=" * 60)
    
    qer = QuantumEfficiencyRatio(
        base_period=14,
        use_multiscale=True,
        use_predictive=True,
        confidence_threshold=0.65
    )
    
    # 取引シミュレーション
    portfolio_value = 10000  # 初期資金
    position = 0  # ポジション（0: ニュートラル, 1: ロング, -1: ショート）
    trade_log = []
    portfolio_history = []
    
    # データを段階的に処理（リアルタイムシミュレーション）
    for i in range(100, len(data), 5):  # 5時間毎に判定
        current_data = data.iloc[:i+1]
        result = qer.calculate(current_data)
        
        current_price = current_data['close'].iloc[-1]
        current_qer = result.values[-1]
        current_confidence = result.confidence_values[-1]
        current_trend = result.current_trend
        current_regime = result.current_regime
        
        # 取引判定ロジック
        should_trade = False
        new_position = position
        
        if current_confidence >= 0.65 and not np.isnan(current_qer):
            if current_trend == 'up' and current_qer > 0.6 and position <= 0:
                # ロングエントリー
                new_position = 1
                should_trade = True
                action = 'LONG_ENTRY'
                
            elif current_trend == 'down' and current_qer > 0.6 and position >= 0:
                # ショートエントリー
                new_position = -1
                should_trade = True
                action = 'SHORT_ENTRY'
                
            elif current_trend == 'range' and position != 0:
                # ポジションクローズ
                new_position = 0
                should_trade = True
                action = 'CLOSE'
        
        # 取引実行
        if should_trade:
            if position != 0:
                # 前のポジションをクローズ
                if position == 1:  # ロングクローズ
                    profit_pct = (current_price - entry_price) / entry_price
                else:  # ショートクローズ
                    profit_pct = (entry_price - current_price) / entry_price
                
                portfolio_value *= (1 + profit_pct)
                
                trade_log.append({
                    'timestamp': current_data['timestamp'].iloc[-1],
                    'action': f'{action}_CLOSE',
                    'price': current_price,
                    'position': position,
                    'profit_pct': profit_pct * 100,
                    'portfolio_value': portfolio_value,
                    'qer': current_qer,
                    'confidence': current_confidence,
                    'regime': current_regime
                })
            
            if new_position != 0:
                # 新しいポジションをオープン
                entry_price = current_price
                trade_log.append({
                    'timestamp': current_data['timestamp'].iloc[-1],
                    'action': action,
                    'price': current_price,
                    'position': new_position,
                    'profit_pct': 0,
                    'portfolio_value': portfolio_value,
                    'qer': current_qer,
                    'confidence': current_confidence,
                    'regime': current_regime
                })
            
            position = new_position
        
        portfolio_history.append({
            'timestamp': current_data['timestamp'].iloc[-1],
            'portfolio_value': portfolio_value,
            'position': position,
            'price': current_price,
            'qer': current_qer,
            'confidence': current_confidence
        })
    
    # 結果の可視化
    portfolio_df = pd.DataFrame(portfolio_history)
    trade_df = pd.DataFrame(trade_log)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. ポートフォリオ価値の推移
    axes[0].plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                color='green', linewidth=2, label='ポートフォリオ価値')
    axes[0].axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='初期資金')
    
    # 取引ポイントをマーク
    for _, trade in trade_df.iterrows():
        color = 'green' if 'LONG' in trade['action'] else 'red' if 'SHORT' in trade['action'] else 'blue'
        axes[0].scatter(trade['timestamp'], trade['portfolio_value'], 
                       color=color, s=50, alpha=0.8, marker='o')
    
    axes[0].set_title('💰 ポートフォリオ価値の推移', fontweight='bold')
    axes[0].set_ylabel('ポートフォリオ価値 ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 価格とポジション
    axes[1].plot(portfolio_df['timestamp'], portfolio_df['price'], 
                color='black', alpha=0.7, label='価格')
    
    # ポジション表示
    long_positions = portfolio_df[portfolio_df['position'] == 1]
    short_positions = portfolio_df[portfolio_df['position'] == -1]
    
    if not long_positions.empty:
        axes[1].scatter(long_positions['timestamp'], long_positions['price'], 
                       color='green', s=20, alpha=0.6, label='ロングポジション')
    
    if not short_positions.empty:
        axes[1].scatter(short_positions['timestamp'], short_positions['price'], 
                       color='red', s=20, alpha=0.6, label='ショートポジション')
    
    axes[1].set_title('📈 価格とポジション', fontweight='bold')
    axes[1].set_ylabel('価格')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. QERと信頼度
    axes[2].plot(portfolio_df['timestamp'], portfolio_df['qer'], 
                color='blue', alpha=0.8, label='QER値')
    axes[2].plot(portfolio_df['timestamp'], portfolio_df['confidence'], 
                color='orange', alpha=0.8, label='信頼度')
    
    axes[2].axhline(y=0.65, color='red', linestyle='--', alpha=0.7, label='信頼度閾値')
    axes[2].axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='QER閾値')
    
    axes[2].set_title('📊 QER & 信頼度', fontweight='bold')
    axes[2].set_ylabel('値')
    axes[2].set_xlabel('時間')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 日付フォーマット調整
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('examples/output/quantum_er_trading_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # パフォーマンス統計
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    total_return = (final_value - 10000) / 10000 * 100
    total_trades = len(trade_df)
    profitable_trades = len([t for t in trade_log if t.get('profit_pct', 0) > 0])
    
    print(f"\n📊 取引パフォーマンス統計")
    print(f"初期資金:           $10,000")
    print(f"最終価値:           ${final_value:,.2f}")
    print(f"総リターン:         {total_return:.2f}%")
    print(f"総取引数:           {total_trades}")
    print(f"勝率:               {(profitable_trades/max(1, total_trades-total_trades//2)*100):.1f}%")
    
    if len(trade_df) > 0:
        print(f"\n🎯 QER統計")
        avg_qer = trade_df['qer'].mean()
        avg_confidence = trade_df['confidence'].mean()
        print(f"平均QER値:          {avg_qer:.3f}")
        print(f"平均信頼度:         {avg_confidence:.3f}")


def main():
    """
    メイン実行関数
    """
    print("🚀 Quantum Efficiency Ratio (QER) デモンストレーション")
    print("=" * 80)
    
    # 出力ディレクトリの作成
    os.makedirs('examples/output', exist_ok=True)
    
    # テストデータの生成
    print("📊 テストデータを生成中...")
    data = generate_synthetic_data(1000)
    
    # 1. 従来のERとの比較
    compare_traditional_vs_quantum_er(data)
    
    # 2. QERの先進機能デモ
    demonstrate_qer_features(data)
    
    # 3. リアルタイム取引シミュレーション
    realtime_trading_simulation(data)
    
    print("\n✅ 全てのデモンストレーションが完了しました！")
    print("📁 結果は examples/output/ ディレクトリに保存されています。")
    
    # QERの主要機能説明
    print("\n🌟 Quantum Efficiency Ratio の革新的機能:")
    print("   1. マルチスケール解析 - 複数時間枠での効率性統合")
    print("   2. 適応的ノイズフィルタ - インテリジェントなノイズ除去")
    print("   3. 予測的成分 - 将来の効率性を先読み")
    print("   4. フラクタル適応 - 市場のフラクタル特性に動的対応")
    print("   5. カスケード型スムージング - 超低遅延での平滑化")
    print("   6. 市場レジーム適応 - トレンド/レンジ/ブレイクアウト自動検出")
    print("   7. 量子的重ね合わせ - 確率的効率性評価")
    print("   8. 信頼度加重 - 計算結果の信頼性を定量化")


if __name__ == "__main__":
    main()