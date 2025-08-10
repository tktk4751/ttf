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
    from indicators.smoother.kalman import Kalman
    from indicators.smoother.multivariate_kalman import MultivariateKalman
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
    
    # 基準価格の生成（ランダムウォーク + トレンド）
    base_price = 100.0
    trend = 0.002  # 上昇トレンド
    volatility = 0.02
    
    prices = [base_price]
    for i in range(1, length):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # OHLC データの生成
    data = []
    for i in range(length):
        base = prices[i]
        daily_vol = abs(np.random.normal(0, volatility * base * 0.5))
        
        # 高値・安値の生成
        high = base + daily_vol * np.random.uniform(0.3, 1.0)
        low = base - daily_vol * np.random.uniform(0.3, 1.0)
        
        # 始値・終値の生成
        if i == 0:
            open_price = base
        else:
            open_price = data[-1]['close'] + np.random.normal(0, volatility * base * 0.1)
        
        close = base + np.random.normal(0, volatility * base * 0.3)
        
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


def create_kalman_comparison_chart(
    data: pd.DataFrame,
    title: str = "カルマンフィルター比較",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    カルマンフィルター比較チャートを作成
    
    Args:
        data: OHLC データ
        title: チャートタイトル
        save_path: 保存パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
    """
    # カルマンフィルターの初期化
    print("カルマンフィルターを初期化中...")
    
    # 従来のカルマンフィルター（終値ベース）
    kalman_close = Kalman(
        process_noise=1e-5,
        observation_noise=1e-3,
        src_type='close'
    )
    
    # 従来のカルマンフィルター（HLC3ベース）
    kalman_hlc3 = Kalman(
        process_noise=1e-5,
        observation_noise=1e-3,
        src_type='hlc3'
    )
    
    # 多変量カルマンフィルター
    multivariate_kalman = MultivariateKalman(
        process_noise=1e-5,
        observation_noise=1e-3,
        volatility_noise=1e-4
    )
    
    # 計算実行
    print("フィルター計算中...")
    kalman_close_result = kalman_close.calculate(data)
    kalman_hlc3_result = kalman_hlc3.calculate(data)
    multivariate_result = multivariate_kalman.calculate(data)
    
    # 結果の取得
    close_filtered = kalman_close_result.filtered_signal
    hlc3_filtered = kalman_hlc3_result.filtered_signal
    
    multivariate_filtered = multivariate_result.filtered_prices
    multivariate_volatility = multivariate_result.volatility_estimates
    multivariate_ranges = multivariate_result.price_range_estimates
    multivariate_confidence = multivariate_result.confidence_scores
    
    # 比較用の元データ
    close_prices = data['close'].values
    hlc3_prices = ((data['high'] + data['low'] + data['close']) / 3).values
    
    # 日付軸の準備
    dates = data.index
    
    # チャート作成
    plt.style.use('default')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, height_ratios=[3, 2, 2, 1], hspace=0.3, wspace=0.3)
    
    # カラーパレット
    colors = {
        'raw_close': '#E74C3C',
        'raw_hlc3': '#F39C12', 
        'kalman_close': '#3498DB',
        'kalman_hlc3': '#9B59B6',
        'multivariate': '#27AE60',
        'volatility': '#E67E22',
        'confidence': '#1ABC9C'
    }
    
    # === メイン価格チャート ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # 元データ
    ax1.plot(dates, close_prices, color=colors['raw_close'], alpha=0.6, linewidth=1, label='元データ(終値)')
    ax1.plot(dates, hlc3_prices, color=colors['raw_hlc3'], alpha=0.6, linewidth=1, label='元データ(HLC3)')
    
    # フィルター結果
    ax1.plot(dates, close_filtered, color=colors['kalman_close'], linewidth=2, label='従来カルマン(終値)')
    ax1.plot(dates, hlc3_filtered, color=colors['kalman_hlc3'], linewidth=2, label='従来カルマン(HLC3)')
    ax1.plot(dates, multivariate_filtered, color=colors['multivariate'], linewidth=2.5, label='多変量カルマン')
    
    # 多変量カルマンの信頼区間
    upper_bound = multivariate_filtered + multivariate_volatility
    lower_bound = multivariate_filtered - multivariate_volatility
    ax1.fill_between(dates, lower_bound, upper_bound, color=colors['multivariate'], alpha=0.2, label='多変量信頼区間')
    
    ax1.set_title(f'{title} - 価格フィルタリング比較', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 日付軸の設定
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    
    # === 誤差比較（左） ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 各フィルターの誤差計算（元の終値との差）
    error_kalman_close = np.abs(close_filtered - close_prices)
    error_kalman_hlc3 = np.abs(hlc3_filtered - close_prices)
    error_multivariate = np.abs(multivariate_filtered - close_prices)
    
    ax2.plot(dates, error_kalman_close, color=colors['kalman_close'], linewidth=1.5, label='従来(終値)')
    ax2.plot(dates, error_kalman_hlc3, color=colors['kalman_hlc3'], linewidth=1.5, label='従来(HLC3)')
    ax2.plot(dates, error_multivariate, color=colors['multivariate'], linewidth=1.5, label='多変量')
    
    ax2.set_title('フィルタリング誤差比較', fontsize=12, fontweight='bold')
    ax2.set_ylabel('絶対誤差', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === ボラティリティ推定（右） ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 実際のボラティリティ（高値-安値）
    actual_volatility = data['high'] - data['low']
    
    ax3.plot(dates, actual_volatility, color='gray', alpha=0.7, linewidth=1, label='実際のレンジ')
    ax3.plot(dates, multivariate_volatility, color=colors['volatility'], linewidth=2, label='推定ボラティリティ')
    ax3.plot(dates, multivariate_ranges, color=colors['multivariate'], linewidth=1.5, linestyle='--', label='動的価格レンジ')
    
    ax3.set_title('ボラティリティ・レンジ推定', fontsize=12, fontweight='bold')
    ax3.set_ylabel('ボラティリティ', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 速度比較（左） ===
    ax4 = fig.add_subplot(gs[2, 0])
    
    # 速度の取得
    kalman_close_velocity = kalman_close_result.state_estimates[:, 1] if kalman_close_result.state_estimates.shape[1] > 1 else np.zeros(len(dates))
    kalman_hlc3_velocity = kalman_hlc3_result.state_estimates[:, 1] if kalman_hlc3_result.state_estimates.shape[1] > 1 else np.zeros(len(dates))
    multivariate_velocity = multivariate_result.velocity_estimates
    
    ax4.plot(dates, kalman_close_velocity, color=colors['kalman_close'], linewidth=1.5, label='従来(終値)')
    ax4.plot(dates, kalman_hlc3_velocity, color=colors['kalman_hlc3'], linewidth=1.5, label='従来(HLC3)')
    ax4.plot(dates, multivariate_velocity, color=colors['multivariate'], linewidth=2, label='多変量')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax4.set_title('速度推定比較', fontsize=12, fontweight='bold')
    ax4.set_ylabel('価格速度', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 信頼度スコア（右） ===
    ax5 = fig.add_subplot(gs[2, 1])
    
    # 信頼度の取得
    kalman_close_confidence = kalman_close_result.confidence_score
    kalman_hlc3_confidence = kalman_hlc3_result.confidence_score
    
    ax5.plot(dates, kalman_close_confidence, color=colors['kalman_close'], linewidth=1.5, label='従来(終値)')
    ax5.plot(dates, kalman_hlc3_confidence, color=colors['kalman_hlc3'], linewidth=1.5, label='従来(HLC3)')
    ax5.plot(dates, multivariate_confidence, color=colors['confidence'], linewidth=2, label='多変量')
    
    ax5.set_title('信頼度スコア比較', fontsize=12, fontweight='bold')
    ax5.set_ylabel('信頼度 (0-1)', fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 統計情報表示 ===
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # 統計計算
    stats_text = f"""
統計情報 (データ期間: {len(dates)}ポイント):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

平均絶対誤差(MAE):  従来(終値): {np.mean(error_kalman_close):.6f}  |  従来(HLC3): {np.mean(error_kalman_hlc3):.6f}  |  多変量: {np.mean(error_multivariate):.6f}

二乗平均平方根誤差(RMSE):  従来(終値): {np.sqrt(np.mean(error_kalman_close**2)):.6f}  |  従来(HLC3): {np.sqrt(np.mean(error_kalman_hlc3**2)):.6f}  |  多変量: {np.sqrt(np.mean(error_multivariate**2)):.6f}

平均信頼度:  従来(終値): {np.mean(kalman_close_confidence):.4f}  |  従来(HLC3): {np.mean(kalman_hlc3_confidence):.4f}  |  多変量: {np.mean(multivariate_confidence):.4f}

平均ボラティリティ推定:  多変量: {np.mean(multivariate_volatility):.6f}  |  実際のレンジ: {np.mean(actual_volatility):.6f}
    """.strip()
    
    ax6.text(0.02, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # 全体タイトル
    fig.suptitle(f'{title}\n従来カルマンフィルター vs 多変量カルマンフィルター', 
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
    parser = argparse.ArgumentParser(description='カルマンフィルター比較チャートの生成')
    parser.add_argument('--config', '-c', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--symbol', '-s', default='SOLUSDT', help='シンボル（リアルデータ使用時）')
    parser.add_argument('--interval', '-i', default='4h', help='時間軸')
    parser.add_argument('--limit', '-l', type=int, default=500, help='データポイント数')
    parser.add_argument('--sample', action='store_true', help='サンプルデータを使用')
    parser.add_argument('--save', help='保存パス（指定時は画像保存）')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 12], help='図のサイズ [幅, 高さ]')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("カルマンフィルター比較チャート生成")
    print("=" * 60)
    
    try:
        if args.sample:
            print("サンプルデータを生成中...")
            data = generate_sample_data(args.limit)
            title = f"カルマンフィルター比較 (サンプルデータ)"
        else:
            print(f"リアルデータを取得中: {args.symbol} ({args.interval})")
            config = load_config(args.config)
            
            data_source = BinanceDataSource(config)
            data = data_source.fetch_data(args.symbol, args.interval, args.limit)
            
            if data is None or len(data) == 0:
                print("データの取得に失敗しました。サンプルデータを使用します。")
                data = generate_sample_data(args.limit)
                title = f"カルマンフィルター比較 (サンプルデータ)"
            else:
                title = f"カルマンフィルター比較 ({args.symbol} {args.interval})"
        
        print(f"データポイント数: {len(data)}")
        print(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        
        # チャート生成
        fig = create_kalman_comparison_chart(
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