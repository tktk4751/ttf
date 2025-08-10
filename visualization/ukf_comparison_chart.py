#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import argparse
import os
import sys
from typing import Optional, Tuple

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.smoother.unscented_kalman_filter_v2 import UnscentedKalmanFilterV2Wrapper
    from data.binance_data_source import BinanceDataSource
    import yaml
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


def load_config(config_path: str = "config.yaml") -> dict:
    """設定ファイルを読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}


def generate_sample_data(length: int = 500) -> pd.DataFrame:
    """
    サンプルOHLCデータを生成する
    
    Args:
        length: データポイント数
        
    Returns:
        OHLC DataFrame
    """
    np.random.seed(42)  # 再現性のため
    
    # 非線形トレンドとボラティリティクラスタリングを含むデータ生成
    base_price = 100.0
    trend = 0.001  # 弱いトレンド
    
    prices = [base_price]
    volatilities = [0.02]  # 初期ボラティリティ
    
    for i in range(1, length):
        # ボラティリティクラスタリング（GARCH風）
        vol_persistence = 0.9
        vol_innovation = 0.1
        new_vol = vol_persistence * volatilities[-1] + vol_innovation * abs(np.random.normal(0, 0.01))
        volatilities.append(max(0.005, min(0.1, new_vol)))  # ボラティリティの制限
        
        # 非線形価格変動
        # トレンド成分
        trend_component = trend * (1 + 0.5 * np.sin(i * 0.01))
        
        # ランダム成分（現在のボラティリティ依存）
        random_component = np.random.normal(0, volatilities[-1])
        
        # レジーム変化（時々大きな変動）
        if np.random.random() < 0.05:  # 5%の確率で大きな変動
            shock = np.random.normal(0, volatilities[-1] * 3)
        else:
            shock = 0
        
        # 価格更新
        price_change = trend_component + random_component + shock
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    volatilities = np.array(volatilities)
    
    # OHLC データの生成
    data = []
    for i in range(length):
        base = prices[i]
        daily_vol = volatilities[i] * base
        
        # 高値・安値の生成（よりリアルに）
        intraday_range = daily_vol * np.random.uniform(0.5, 2.0)
        high = base + intraday_range * np.random.uniform(0.3, 1.0)
        low = base - intraday_range * np.random.uniform(0.3, 1.0)
        
        # 始値・終値の生成
        if i == 0:
            open_price = base
        else:
            # 前日終値からのギャップ
            gap = np.random.normal(0, daily_vol * 0.2)
            open_price = data[-1]['close'] + gap
        
        # 終値は基準価格周辺
        close = base + np.random.normal(0, daily_vol * 0.3)
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    # タイムスタンプの追加
    start_date = datetime.now() - timedelta(days=length)
    df['timestamp'] = [start_date + timedelta(hours=i) for i in range(length)]
    df.set_index('timestamp', inplace=True)
    
    return df


def create_ukf_comparison_chart(
    data: pd.DataFrame,
    title: str = "UKF比較",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14)
):
    """
    UKF比較チャートを作成
    
    Args:
        data: OHLC データ
        title: チャートタイトル
        save_path: 保存パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
    """
    # UKFフィルターの初期化
    print("UKFフィルターを初期化中...")
    
    # UKF V1（既存実装）
    ukf_v1 = UnscentedKalmanFilter(
        src_type='close',
        alpha=0.1,
        beta=2.0,
        kappa=0.0,
        process_noise_scale=0.01,
        volatility_window=10,
        adaptive_noise=True
    )
    
    # UKF V2（アカデミック実装）
    ukf_v2 = UnscentedKalmanFilterV2Wrapper(
        src_type='close',
        kappa=0.0,
        process_noise_scale=0.01,
        observation_noise_scale=0.001,
        max_steps=1000
    )
    
    # 計算実行
    print("フィルター計算中...")
    ukf_v1_result = ukf_v1.calculate(data)
    ukf_v2_result = ukf_v2.calculate(data)
    
    # 結果の取得
    close_prices = data['close'].values
    
    # UKF V1の結果
    ukf_v1_filtered = ukf_v1_result.filtered_values
    ukf_v1_velocity = ukf_v1_result.velocity_estimates
    ukf_v1_acceleration = ukf_v1_result.acceleration_estimates
    ukf_v1_uncertainty = ukf_v1_result.uncertainty
    ukf_v1_confidence = ukf_v1_result.confidence_scores
    
    # UKF V2の結果
    ukf_v2_filtered = ukf_v2_result.filtered_values
    ukf_v2_velocity = ukf_v2_result.state_estimates[:, 1] if ukf_v2_result.state_estimates.shape[1] > 1 else np.zeros(len(close_prices))
    ukf_v2_acceleration = ukf_v2_result.state_estimates[:, 2] if ukf_v2_result.state_estimates.shape[1] > 2 else np.zeros(len(close_prices))
    ukf_v2_confidence = ukf_v2_result.confidence_scores
    ukf_v2_uncertainty = np.sqrt(ukf_v2_result.error_covariance[:, 0])  # 価格の不確実性
    
    # 日付軸の準備
    dates = data.index
    
    # チャート作成
    plt.style.use('default')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 2, height_ratios=[3, 2, 2, 2, 1], hspace=0.3, wspace=0.3)
    
    # カラーパレット
    colors = {
        'raw': '#E74C3C',
        'ukf_v1': '#3498DB',
        'ukf_v2': '#27AE60',
        'uncertainty': '#F39C12',
        'confidence': '#9B59B6',
        'velocity': '#E67E22',
        'acceleration': '#1ABC9C'
    }
    
    # === メイン価格チャート ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # 元データ
    ax1.plot(dates, close_prices, color=colors['raw'], alpha=0.7, linewidth=1, label='元データ(終値)', zorder=1)
    
    # フィルター結果
    ax1.plot(dates, ukf_v1_filtered, color=colors['ukf_v1'], linewidth=2, label='UKF V1 (既存実装)', zorder=3)
    ax1.plot(dates, ukf_v2_filtered, color=colors['ukf_v2'], linewidth=2.5, label='UKF V2 (アカデミック)', zorder=4)
    
    # 不確実性の信頼区間（UKF V1）
    upper_v1 = ukf_v1_filtered + ukf_v1_uncertainty
    lower_v1 = ukf_v1_filtered - ukf_v1_uncertainty
    ax1.fill_between(dates, lower_v1, upper_v1, color=colors['ukf_v1'], alpha=0.2, label='UKF V1 信頼区間', zorder=2)
    
    # 不確実性の信頼区間（UKF V2）
    upper_v2 = ukf_v2_filtered + ukf_v2_uncertainty
    lower_v2 = ukf_v2_filtered - ukf_v2_uncertainty
    ax1.fill_between(dates, lower_v2, upper_v2, color=colors['ukf_v2'], alpha=0.2, label='UKF V2 信頼区間', zorder=2)
    
    ax1.set_title(f'{title} - フィルタリング比較', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    
    # === 誤差比較（左） ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 各フィルターの誤差計算
    error_v1 = np.abs(ukf_v1_filtered - close_prices)
    error_v2 = np.abs(ukf_v2_filtered - close_prices)
    
    ax2.plot(dates, error_v1, color=colors['ukf_v1'], linewidth=1.5, label='UKF V1')
    ax2.plot(dates, error_v2, color=colors['ukf_v2'], linewidth=1.5, label='UKF V2')
    
    ax2.set_title('フィルタリング誤差比較', fontsize=12, fontweight='bold')
    ax2.set_ylabel('絶対誤差', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 不確実性比較（右） ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.plot(dates, ukf_v1_uncertainty, color=colors['ukf_v1'], linewidth=1.5, label='UKF V1')
    ax3.plot(dates, ukf_v2_uncertainty, color=colors['ukf_v2'], linewidth=1.5, label='UKF V2')
    
    ax3.set_title('不確実性比較', fontsize=12, fontweight='bold')
    ax3.set_ylabel('不確実性', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 速度推定比較（左） ===
    ax4 = fig.add_subplot(gs[2, 0])
    
    ax4.plot(dates, ukf_v1_velocity, color=colors['ukf_v1'], linewidth=1.5, label='UKF V1')
    ax4.plot(dates, ukf_v2_velocity, color=colors['ukf_v2'], linewidth=1.5, label='UKF V2')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax4.set_title('速度推定比較', fontsize=12, fontweight='bold')
    ax4.set_ylabel('価格速度', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 加速度推定比較（右） ===
    ax5 = fig.add_subplot(gs[2, 1])
    
    ax5.plot(dates, ukf_v1_acceleration, color=colors['ukf_v1'], linewidth=1.5, label='UKF V1')
    ax5.plot(dates, ukf_v2_acceleration, color=colors['ukf_v2'], linewidth=1.5, label='UKF V2')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax5.set_title('加速度推定比較', fontsize=12, fontweight='bold')
    ax5.set_ylabel('価格加速度', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 信頼度比較（左） ===
    ax6 = fig.add_subplot(gs[3, 0])
    
    ax6.plot(dates, ukf_v1_confidence, color=colors['ukf_v1'], linewidth=1.5, label='UKF V1')
    ax6.plot(dates, ukf_v2_confidence, color=colors['ukf_v2'], linewidth=1.5, label='UKF V2')
    
    ax6.set_title('信頼度スコア比較', fontsize=12, fontweight='bold')
    ax6.set_ylabel('信頼度 (0-1)', fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === イノベーション比較（右） ===
    ax7 = fig.add_subplot(gs[3, 1])
    
    ukf_v1_innovations = ukf_v1_result.innovations
    ukf_v2_innovations = ukf_v2_result.innovations
    
    ax7.plot(dates, ukf_v1_innovations, color=colors['ukf_v1'], linewidth=1, alpha=0.7, label='UKF V1')
    ax7.plot(dates, ukf_v2_innovations, color=colors['ukf_v2'], linewidth=1, alpha=0.7, label='UKF V2')
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax7.set_title('イノベーション比較', fontsize=12, fontweight='bold')
    ax7.set_ylabel('イノベーション', fontsize=10)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 統計情報表示 ===
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')
    
    # 統計計算
    stats_text = f"""
統計情報 (データ期間: {len(dates)}ポイント):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

平均絶対誤差(MAE):  UKF V1: {np.mean(error_v1):.6f}  |  UKF V2: {np.mean(error_v2):.6f}

二乗平均平方根誤差(RMSE):  UKF V1: {np.sqrt(np.mean(error_v1**2)):.6f}  |  UKF V2: {np.sqrt(np.mean(error_v2**2)):.6f}

平均信頼度:  UKF V1: {np.mean(ukf_v1_confidence):.4f}  |  UKF V2: {np.mean(ukf_v2_confidence):.4f}

平均不確実性:  UKF V1: {np.mean(ukf_v1_uncertainty):.6f}  |  UKF V2: {np.mean(ukf_v2_uncertainty):.6f}

速度推定範囲:  UKF V1: [{np.min(ukf_v1_velocity):.4f}, {np.max(ukf_v1_velocity):.4f}]  |  UKF V2: [{np.min(ukf_v2_velocity):.4f}, {np.max(ukf_v2_velocity):.4f}]
    """.strip()
    
    ax8.text(0.02, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax8.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # 全体タイトル
    fig.suptitle(f'{title}\nUKF V1 (既存実装) vs UKF V2 (アカデミック実装)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"チャートを保存しました: {save_path}")
    else:
        plt.show()
    
    return fig


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='UKF比較チャートの生成')
    parser.add_argument('--config', '-c', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--symbol', '-s', default='SOLUSDT', help='シンボル（リアルデータ使用時）')
    parser.add_argument('--interval', '-i', default='4h', help='時間軸')
    parser.add_argument('--limit', '-l', type=int, default=500, help='データポイント数')
    parser.add_argument('--sample', action='store_true', help='サンプルデータを使用')
    parser.add_argument('--save', help='保存パス（指定時は画像保存）')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 14], help='図のサイズ [幅, 高さ]')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UKF比較チャート生成")
    print("=" * 60)
    
    try:
        if args.sample:
            print("サンプルデータを生成中...")
            data = generate_sample_data(args.limit)
            title = f"UKF比較 (サンプルデータ)"
        else:
            print(f"リアルデータを取得中: {args.symbol} ({args.interval})")
            config = load_config(args.config)
            
            # BinanceDataSourceのbase_dirを設定から取得、デフォルト値を設定
            base_dir = "data/binance"
            if config and 'data' in config and 'base_dir' in config['data']:
                base_dir = config['data']['base_dir']
            
            data_source = BinanceDataSource(base_dir)
            data = data_source.load_data(args.symbol, args.interval)
            
            # データポイント数制限
            if len(data) > args.limit:
                data = data.tail(args.limit)
            
            if data is None or len(data) == 0:
                print("データの取得に失敗しました。サンプルデータを使用します。")
                data = generate_sample_data(args.limit)
                title = f"UKF比較 (サンプルデータ)"
            else:
                title = f"UKF比較 ({args.symbol} {args.interval})"
        
        print(f"データポイント数: {len(data)}")
        print(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        
        # チャート生成
        fig = create_ukf_comparison_chart(
            data=data,
            title=title,
            save_path=args.save,
            figsize=tuple(args.figsize)
        )
        
        print("処理が完了しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())