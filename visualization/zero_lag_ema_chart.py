#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ゼロラグEMA比較チャートの生成

標準EMA、ZLEMA、価格データを比較表示する可視化ツール。
"""

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
    from indicators.smoother.zero_lag_ema import ZeroLagEMA, zlema, fast_zlema
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


def calculate_standard_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """標準EMAを計算"""
    length = len(prices)
    if length < period:
        return np.full(length, np.nan)
    
    alpha = 2.0 / (period + 1.0)
    ema = np.zeros(length)
    
    # 初期値（SMA）
    ema[:period-1] = np.nan
    ema[period-1] = np.mean(prices[:period])
    
    # EMA計算
    for i in range(period, length):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i-1]
    
    return ema


def generate_sample_data(length: int = 500) -> pd.DataFrame:
    """
    サンプルOHLCデータを生成する
    
    Args:
        length: データポイント数
        
    Returns:
        OHLC DataFrame
    """
    np.random.seed(42)  # 再現性のため
    
    # 非線形トレンドとボラティリティを含むデータ生成
    base_price = 100.0
    trend = 0.001
    
    prices = [base_price]
    volatilities = [0.02]
    
    for i in range(1, length):
        # ボラティリティクラスタリング
        vol_persistence = 0.9
        vol_innovation = 0.1
        new_vol = vol_persistence * volatilities[-1] + vol_innovation * abs(np.random.normal(0, 0.01))
        volatilities.append(max(0.005, min(0.1, new_vol)))
        
        # 非線形価格変動
        trend_component = trend * (1 + 0.5 * np.sin(i * 0.01))
        random_component = np.random.normal(0, volatilities[-1])
        
        # 時々大きな変動
        if np.random.random() < 0.05:
            shock = np.random.normal(0, volatilities[-1] * 3)
        else:
            shock = 0
        
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
        
        intraday_range = daily_vol * np.random.uniform(0.5, 2.0)
        high = base + intraday_range * np.random.uniform(0.3, 1.0)
        low = base - intraday_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = base
        else:
            gap = np.random.normal(0, daily_vol * 0.2)
            open_price = data[-1]['close'] + gap
        
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


def create_zlema_comparison_chart(
    data: pd.DataFrame,
    periods: list = [14, 21, 50],
    title: str = "ZLEMA比較",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    ZLEMA比較チャートを作成
    
    Args:
        data: OHLC データ
        periods: 比較する期間のリスト
        title: チャートタイトル
        save_path: 保存パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
    """
    print("ZLEMAインディケーターを初期化中...")
    
    # 価格データ
    close_prices = data['close'].values
    dates = data.index
    
    # 各期間でのZLEMAと標準EMAを計算
    results = {}
    for period in periods:
        print(f"期間 {period} の計算中...")
        
        # ZLEMA計算
        zlema_indicator = ZeroLagEMA(period=period, src_type='close')
        zlema_result = zlema_indicator.calculate(data)
        
        # 標準EMA計算
        standard_ema = calculate_standard_ema(close_prices, period)
        
        results[period] = {
            'zlema': zlema_result.values,
            'ema': standard_ema,
            'lag_reduced': zlema_result.lag_reduced_data
        }
    
    # チャート作成
    plt.style.use('default')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, height_ratios=[3, 2, 2, 1], hspace=0.3, wspace=0.3)
    
    # カラーパレット
    colors = {
        'price': '#E74C3C',
        'zlema_14': '#3498DB',
        'zlema_21': '#27AE60', 
        'zlema_50': '#9B59B6',
        'ema_14': '#F39C12',
        'ema_21': '#E67E22',
        'ema_50': '#1ABC9C'
    }
    
    # === メイン価格チャート ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # 価格データ
    ax1.plot(dates, close_prices, color=colors['price'], alpha=0.7, linewidth=1, 
             label='終値', zorder=1)
    
    # ZLEMA
    for i, period in enumerate(periods):
        color_key = f'zlema_{period}'
        if color_key in colors:
            color = colors[color_key]
        else:
            color = plt.cm.Set1(i)
        
        ax1.plot(dates, results[period]['zlema'], color=color, linewidth=2.5, 
                label=f'ZLEMA({period})', zorder=3)
    
    # 標準EMA（薄く表示）
    for i, period in enumerate(periods):
        color_key = f'ema_{period}'
        if color_key in colors:
            color = colors[color_key]
        else:
            color = plt.cm.Set2(i)
        
        ax1.plot(dates, results[period]['ema'], color=color, linewidth=1.5, 
                alpha=0.6, linestyle='--', label=f'EMA({period})', zorder=2)
    
    ax1.set_title(f'{title} - ZLEMA vs 標準EMA比較', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    
    # === 価格差分チャート（左） ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    for i, period in enumerate(periods):
        price_diff = results[period]['zlema'] - close_prices
        color_key = f'zlema_{period}'
        color = colors.get(color_key, plt.cm.Set1(i))
        
        ax2.plot(dates, price_diff, color=color, linewidth=1.5, 
                label=f'ZLEMA({period}) - 価格')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('ZLEMA - 価格差分', fontsize=12, fontweight='bold')
    ax2.set_ylabel('価格差分', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === ZLEMA vs EMA差分チャート（右） ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    for i, period in enumerate(periods):
        ma_diff = results[period]['zlema'] - results[period]['ema']
        color_key = f'zlema_{period}'
        color = colors.get(color_key, plt.cm.Set1(i))
        
        ax3.plot(dates, ma_diff, color=color, linewidth=1.5, 
                label=f'期間{period}')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('ZLEMA - EMA差分', fontsize=12, fontweight='bold')
    ax3.set_ylabel('差分', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === ラグ除去データチャート（左） ===
    ax4 = fig.add_subplot(gs[2, 0])
    
    ax4.plot(dates, close_prices, color=colors['price'], alpha=0.5, linewidth=1, 
             label='元価格')
    
    for i, period in enumerate(periods):
        if not np.all(np.isnan(results[period]['lag_reduced'])):
            color_key = f'zlema_{period}'
            color = colors.get(color_key, plt.cm.Set1(i))
            
            ax4.plot(dates, results[period]['lag_reduced'], color=color, linewidth=1.5,
                    alpha=0.8, label=f'ラグ除去({period})')
    
    ax4.set_title('ラグ除去データ', fontsize=12, fontweight='bold')
    ax4.set_ylabel('価格', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 応答性比較チャート（右） ===
    ax5 = fig.add_subplot(gs[2, 1])
    
    # 価格変化率
    price_returns = np.diff(close_prices) / close_prices[:-1] * 100
    dates_returns = dates[1:]
    
    ax5.plot(dates_returns, price_returns, color=colors['price'], alpha=0.6, 
             linewidth=1, label='価格変化率')
    
    # ZLEMAの応答性（変化率）
    for i, period in enumerate(periods):
        zlema_vals = results[period]['zlema']
        zlema_returns = np.diff(zlema_vals) / zlema_vals[:-1] * 100
        
        color_key = f'zlema_{period}'
        color = colors.get(color_key, plt.cm.Set1(i))
        
        ax5.plot(dates_returns, zlema_returns, color=color, linewidth=1.5,
                alpha=0.8, label=f'ZLEMA({period})')
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('変化率比較 (%)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('変化率 (%)', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 統計情報表示 ===
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # 統計計算
    stats_lines = [f"統計情報 (データ期間: {len(dates)}ポイント):"]
    stats_lines.append("━" * 100)
    stats_lines.append("")
    
    for period in periods:
        zlema_vals = results[period]['zlema']
        ema_vals = results[period]['ema']
        
        # 有効値のみで計算
        valid_mask = ~(np.isnan(zlema_vals) | np.isnan(ema_vals))
        if np.any(valid_mask):
            valid_zlema = zlema_vals[valid_mask]
            valid_ema = ema_vals[valid_mask]
            valid_price = close_prices[valid_mask]
            
            # 統計指標
            zlema_mae = np.mean(np.abs(valid_zlema - valid_price))
            ema_mae = np.mean(np.abs(valid_ema - valid_price))
            responsiveness = np.mean(np.abs(valid_zlema - valid_ema))
            
            stats_lines.append(
                f"期間{period:>2}: ZLEMA MAE={zlema_mae:>8.4f}  |  "
                f"EMA MAE={ema_mae:>8.4f}  |  応答性差={responsiveness:>8.4f}"
            )
    
    stats_text = "\n".join(stats_lines)
    
    ax6.text(0.02, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # 全体タイトル
    fig.suptitle(f'{title}\nゼロラグEMA vs 標準EMA - 応答性と精度の比較', 
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
    parser = argparse.ArgumentParser(description='ゼロラグEMA比較チャートの生成')
    parser.add_argument('--config', '-c', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--symbol', '-s', default='SOLUSDT', help='シンボル（リアルデータ使用時）')
    parser.add_argument('--interval', '-i', default='4h', help='時間軸')
    parser.add_argument('--limit', '-l', type=int, default=500, help='データポイント数')
    parser.add_argument('--sample', action='store_true', help='サンプルデータを使用')
    parser.add_argument('--save', help='保存パス（指定時は画像保存）')
    parser.add_argument('--periods', nargs='+', type=int, default=[14, 21, 50], 
                       help='比較する期間（例: --periods 14 21 50）')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 12], help='図のサイズ [幅, 高さ]')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ゼロラグEMA比較チャート生成")
    print("=" * 60)
    
    try:
        if args.sample:
            print("サンプルデータを生成中...")
            data = generate_sample_data(args.limit)
            title = f"ZLEMA比較 (サンプルデータ)"
        else:
            print(f"リアルデータを取得中: {args.symbol} ({args.interval})")
            config = load_config(args.config)
            
            # BinanceDataSourceのbase_dirを設定から取得
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
                title = f"ZLEMA比較 (サンプルデータ)"
            else:
                title = f"ZLEMA比較 ({args.symbol} {args.interval})"
        
        print(f"データポイント数: {len(data)}")
        print(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        print(f"比較期間: {args.periods}")
        
        # チャート生成
        fig = create_zlema_comparison_chart(
            data=data,
            periods=args.periods,
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