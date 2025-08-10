#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合スムーサー比較チャートの生成

複数のスムーサーを同時に比較表示する可視化ツール。
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
from typing import Optional, Tuple, List

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from indicators.smoother.unified_smoother import UnifiedSmoother, compare_smoothers
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
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


def normalize_smoother_values(values: np.ndarray, target_length: int) -> np.ndarray:
    """スムーサー結果の形状を正規化する"""
    # NumPy配列に変換
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    
    # スカラー値の場合は同じ値の配列を作成
    if values.ndim == 0:
        return np.full(target_length, values)
    elif values.ndim > 1:
        values = values.flatten()
    
    # データ長の調整
    if len(values) != target_length:
        if len(values) < target_length:
            # 不足分をNaNで埋める
            padded_values = np.full(target_length, np.nan)
            padded_values[-len(values):] = values
            return padded_values
        else:
            # 余分な部分を切り捨て
            return values[-target_length:]
    
    return values


def create_unified_smoother_comparison_chart(
    data: pd.DataFrame,
    smoother_types: List[str] = ['frama', 'super_smoother', 'zero_lag_ema'],
    title: str = "統合スムーサー比較",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    統合スムーサー比較チャートを作成
    
    Args:
        data: OHLC データ
        smoother_types: 比較するスムーサータイプのリスト
        title: チャートタイトル
        save_path: 保存パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
    """
    print("統合スムーサーを初期化中...")
    
    # 価格データ
    close_prices = data['close'].values
    dates = data.index
    
    # 各スムーサーでの計算
    results = {}
    smoothers_info = {}
    
    for smoother_type in smoother_types:
        print(f"{smoother_type} の計算中...")
        
        try:
            smoother = UnifiedSmoother(smoother_type=smoother_type, src_type='close')
            result = smoother.calculate(data)
            
            results[smoother_type] = result
            smoothers_info[smoother_type] = smoother.get_smoother_info()
            
        except Exception as e:
            print(f"{smoother_type}の計算エラー: {e}")
            continue
    
    if not results:
        print("有効なスムーサー結果がありません")
        return None
    
    # チャート作成
    plt.style.use('default')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, height_ratios=[3, 2, 2, 1], hspace=0.3, wspace=0.3)
    
    # カラーパレット
    colors = {
        'price': '#E74C3C',
        'frama': '#3498DB',
        'super_smoother': '#27AE60', 
        'zero_lag_ema': '#9B59B6',
        'ultimate_smoother': '#F39C12',
        'kalman': '#E67E22',
        'multivariate_kalman': '#1ABC9C',
        'adaptive_kalman': '#34495E',
        'ukf': '#8E44AD',
        'ukf_v2': '#D35400'
    }
    
    # === メイン価格チャート ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # 価格データ
    ax1.plot(dates, close_prices, color=colors['price'], alpha=0.7, linewidth=1, 
             label='終値', zorder=1)
    
    # スムーサー結果
    for i, (smoother_type, result) in enumerate(results.items()):
        color = colors.get(smoother_type, plt.cm.Set1(i))
        description = smoothers_info[smoother_type]['description']
        
        # 結果の形状を正規化
        values = normalize_smoother_values(result.values, len(dates))
        
        ax1.plot(dates, values, color=color, linewidth=2, 
                label=f'{smoother_type} ({description[:20]})', zorder=3)
    
    ax1.set_title(f'{title} - スムーサー比較', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    
    # === 価格差分チャート（左） ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    for i, (smoother_type, result) in enumerate(results.items()):
        # 結果の形状を正規化
        values = normalize_smoother_values(result.values, len(close_prices))
        
        price_diff = values - close_prices
        color = colors.get(smoother_type, plt.cm.Set1(i))
        
        ax2.plot(dates, price_diff, color=color, linewidth=1.5, 
                label=f'{smoother_type}')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('スムーサー - 価格差分', fontsize=12, fontweight='bold')
    ax2.set_ylabel('価格差分', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === スムージング効果チャート（右） ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 価格のボラティリティ（移動標準偏差）
    window = 20
    price_vol = pd.Series(close_prices).rolling(window=window).std().values
    
    ax3.plot(dates, price_vol, color=colors['price'], alpha=0.7, linewidth=1, 
             label='価格ボラティリティ')
    
    # 各スムーサーのボラティリティ
    for i, (smoother_type, result) in enumerate(results.items()):
        # 結果の形状を正規化
        values = normalize_smoother_values(result.values, len(dates))
        
        smoother_vol = pd.Series(values).rolling(window=window).std().values
        color = colors.get(smoother_type, plt.cm.Set1(i))
        
        ax3.plot(dates, smoother_vol, color=color, linewidth=1.5, 
                label=f'{smoother_type}')
    
    ax3.set_title('ボラティリティ比較', fontsize=12, fontweight='bold')
    ax3.set_ylabel('移動標準偏差', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 応答性比較チャート（左） ===
    ax4 = fig.add_subplot(gs[2, 0])
    
    # 価格変化率
    price_returns = np.diff(close_prices) / close_prices[:-1] * 100
    dates_returns = dates[1:]
    
    ax4.plot(dates_returns, price_returns, color=colors['price'], alpha=0.6, 
             linewidth=1, label='価格変化率')
    
    # スムーサーの応答性（変化率）
    for i, (smoother_type, result) in enumerate(results.items()):
        # 結果の形状を正規化
        values = normalize_smoother_values(result.values, len(close_prices))
        
        # ゼロ除算を避けるための処理
        valid_mask = (values[:-1] != 0) & (~np.isnan(values[:-1])) & (~np.isnan(values[1:]))
        if np.any(valid_mask):
            smoother_returns = np.full(len(values) - 1, np.nan)
            smoother_returns[valid_mask] = np.diff(values)[valid_mask] / values[:-1][valid_mask] * 100
        else:
            smoother_returns = np.full(len(values) - 1, np.nan)
        
        color = colors.get(smoother_type, plt.cm.Set1(i))
        
        ax4.plot(dates_returns, smoother_returns, color=color, linewidth=1.5,
                alpha=0.8, label=f'{smoother_type}')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('変化率比較 (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('変化率 (%)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 誤差比較チャート（右） ===
    ax5 = fig.add_subplot(gs[2, 1])
    
    # 各スムーサーの絶対誤差
    for i, (smoother_type, result) in enumerate(results.items()):
        # 結果の形状を正規化
        values = normalize_smoother_values(result.values, len(close_prices))
        
        abs_error = np.abs(values - close_prices)
        color = colors.get(smoother_type, plt.cm.Set1(i))
        
        ax5.plot(dates, abs_error, color=color, linewidth=1.5,
                alpha=0.8, label=f'{smoother_type}')
    
    ax5.set_title('絶対誤差比較', fontsize=12, fontweight='bold')
    ax5.set_ylabel('絶対誤差', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # === 統計情報表示 ===
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # 統計計算
    stats_lines = [f"統計情報 (データ期間: {len(dates)}ポイント):"]
    stats_lines.append("━" * 120)
    stats_lines.append("")
    
    for smoother_type, result in results.items():
        # 結果の形状を正規化
        smoother_vals = normalize_smoother_values(result.values, len(close_prices))
        
        # 有効値のみで計算
        valid_mask = ~np.isnan(smoother_vals) & ~np.isnan(close_prices)
        if np.any(valid_mask):
            valid_smoother = smoother_vals[valid_mask]
            valid_price = close_prices[valid_mask]
            
            # 統計指標
            mae = np.mean(np.abs(valid_smoother - valid_price))
            rmse = np.sqrt(np.mean((valid_smoother - valid_price)**2))
            
            # 相関の計算（少なくとも2個の値が必要）
            if len(valid_smoother) >= 2:
                correlation = np.corrcoef(valid_smoother, valid_price)[0, 1]
                smoothness = np.std(np.diff(valid_smoother)) / np.std(np.diff(valid_price))
            else:
                correlation = np.nan
                smoothness = np.nan
            
            info = smoothers_info[smoother_type]
            
            stats_lines.append(
                f"{smoother_type:<18}: MAE={mae:>7.4f}  |  RMSE={rmse:>7.4f}  |  "
                f"相関={correlation:>6.4f}  |  平滑度={smoothness:>6.4f}  |  {info['description']}"
            )
    
    stats_text = "\n".join(stats_lines)
    
    ax6.text(0.02, 0.5, stats_text, fontsize=9, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # 全体タイトル
    fig.suptitle(f'{title}\n統合スムーサー比較 - 精度・応答性・平滑性の分析', 
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
    parser = argparse.ArgumentParser(description='統合スムーサー比較チャートの生成')
    parser.add_argument('--config', '-c', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--symbol', '-s', default='SOLUSDT', help='シンボル（リアルデータ使用時）')
    parser.add_argument('--interval', '-i', default='4h', help='時間軸')
    parser.add_argument('--limit', '-l', type=int, default=500, help='データポイント数')
    parser.add_argument('--sample', action='store_true', help='サンプルデータを使用')
    parser.add_argument('--save', help='保存パス（指定時は画像保存）')
    parser.add_argument('--smoothers', nargs='+', 
                       default=['frama', 'super_smoother', 'zero_lag_ema'], 
                       help='比較するスムーサー（例: --smoothers frama zero_lag_ema）')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 12], help='図のサイズ [幅, 高さ]')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("統合スムーサー比較チャート生成")
    print("=" * 60)
    
    try:
        if args.sample:
            print("サンプルデータを生成中...")
            data = generate_sample_data(args.limit)
            title = f"統合スムーサー比較 (サンプルデータ)"
        else:
            print(f"リアルデータを取得中: {args.symbol} ({args.interval})")
            config = load_config(args.config)
            
            try:
                # z_adaptive_trend_chart.pyと同じ方法でデータを取得
                binance_config = config.get('binance_data', {})
                data_dir = binance_config.get('data_dir', 'data/binance')
                binance_data_source = BinanceDataSource(data_dir)
                
                # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
                dummy_csv_source = CSVDataSource("dummy")
                data_loader = DataLoader(
                    data_source=dummy_csv_source,
                    binance_data_source=binance_data_source
                )
                data_processor = DataProcessor()
                
                # データの読み込みと処理
                print("データを読み込み・処理中...")
                raw_data = data_loader.load_data_from_config(config)
                processed_data = {
                    symbol: data_processor.process(df)
                    for symbol, df in raw_data.items()
                }
                
                # 最初のシンボルのデータを取得
                if processed_data:
                    first_symbol = next(iter(processed_data))
                    data = processed_data[first_symbol]
                    
                    # データポイント数制限
                    if len(data) > args.limit:
                        data = data.tail(args.limit)
                    
                    print(f"データ読み込み完了: {first_symbol}")
                    print(f"期間: {data.index.min()} → {data.index.max()}")
                    print(f"データ数: {len(data)}")
                    
                    title = f"統合スムーサー比較 ({first_symbol} {args.interval})"
                else:
                    raise ValueError("設定ファイルからデータを取得できませんでした")
                    
            except Exception as e:
                print(f"リアルデータ取得エラー: {e}")
                print("サンプルデータを使用します。")
                data = generate_sample_data(args.limit)
                title = f"統合スムーサー比較 (サンプルデータ)"
        
        print(f"データポイント数: {len(data)}")
        print(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        print(f"比較スムーサー: {args.smoothers}")
        
        # チャート生成
        fig = create_unified_smoother_comparison_chart(
            data=data,
            smoother_types=args.smoothers,
            title=title,
            save_path=args.save,
            figsize=tuple(args.figsize)
        )
        
        if fig:
            print("処理が完了しました！")
        else:
            print("チャート生成に失敗しました。")
            return 1
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())