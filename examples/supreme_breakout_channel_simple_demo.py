#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **Supreme Breakout Channel Simple Demo** 🚀

人類史上最強ブレイクアウトチャネルの簡単なデモ実装
- 基本的な可視化機能
- エラー処理強化版
- 初心者向けの使いやすいインターフェース
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

# プロジェクトのルートディレクトリを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from indicators.supreme_breakout_channel import SupremeBreakoutChannel
    from data.data_loader import DataLoader, CSVDataSource
    from data.binance_data_source import BinanceDataSource
    from data.data_processor import DataProcessor
    import yaml
except ImportError as e:
    print(f"Import Error: {e}")
    print("ダミーデータでデモを実行します...")

def create_dummy_market_data(days=365, timeframe_hours=4):
    """
    リアルな価格動作を模擬したダミーマーケットデータを生成
    """
    np.random.seed(42)
    periods = days * 24 // timeframe_hours
    dates = pd.date_range(start='2024-01-01', periods=periods, freq=f'{timeframe_hours}H')
    
    # ビットコインらしい価格動作を生成
    initial_price = 45000
    price_data = []
    current_price = initial_price
    
    for i in range(periods):
        # トレンド + ノイズ + 周期性
        trend = 0.001 * np.sin(i * 0.01)  # 長期トレンド
        cycle = 0.02 * np.sin(i * 0.1)   # 中期サイクル
        noise = np.random.normal(0, 0.015)  # ランダムノイズ
        
        # ボラティリティクラスター
        if i % 100 < 20:  # 20%の期間で高ボラティリティ
            noise *= 2
        
        change = trend + cycle + noise
        current_price *= (1 + change)
        
        # OHLC生成
        high = current_price * (1 + abs(np.random.normal(0, 0.008)))
        low = current_price * (1 - abs(np.random.normal(0, 0.008)))
        open_price = current_price + np.random.normal(0, current_price * 0.005)
        volume = np.random.uniform(500, 2000)
        
        price_data.append({
            'datetime': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    return pd.DataFrame(price_data).set_index('datetime')

def plot_supreme_breakout_channel(data, title="Supreme Breakout Channel Demo"):
    """
    Supreme Breakout Channelの基本的な可視化
    """
    print("🚀 Supreme Breakout Channel計算中...")
    
    # Supreme Breakout Channelインジケーターを初期化
    sbc = SupremeBreakoutChannel(
        atr_period=14,
        base_multiplier=2.0,
        kalman_process_noise=0.01,
        min_strength_threshold=0.2,  # より緩い設定
        min_confidence_threshold=0.25,  # より緩い設定
        src_type='hlc3'
    )
    
    try:
        # 計算実行
        result = sbc.calculate(data)
        
        if result is None:
            print("❌ Supreme Breakout Channel計算に失敗しました")
            return None
        
        print(f"✅ 計算完了 - データ数: {len(result.upper_channel)}")
        
        # NaN値をチェック
        upper_nan = np.isnan(result.upper_channel).sum()
        lower_nan = np.isnan(result.lower_channel).sum()
        center_nan = np.isnan(result.centerline).sum()
        
        print(f"📊 NaN値 - 上限: {upper_nan}, 下限: {lower_nan}, 中心: {center_nan}")
        
        # 有効なデータがあるかチェック
        valid_data = ~(np.isnan(result.upper_channel) | np.isnan(result.lower_channel) | np.isnan(result.centerline))
        valid_count = np.sum(valid_data)
        
        print(f"📈 有効データ数: {valid_count}/{len(result.upper_channel)}")
        
        if valid_count < 50:
            print("⚠️  有効データが少なすぎます。ダミーチャネルを生成します。")
            # 簡易チャネル生成
            src_prices = (data['high'] + data['low'] + data['close']) / 3
            sma = src_prices.rolling(20).mean()
            std = src_prices.rolling(20).std()
            
            upper_ch = sma + 2 * std
            lower_ch = sma - 2 * std
            center_ch = sma
            
            # データ長を合わせる
            if len(upper_ch) != len(result.upper_channel):
                padding = len(result.upper_channel) - len(upper_ch)
                if padding > 0:
                    upper_ch = pd.concat([pd.Series([np.nan] * padding), upper_ch]).values
                    lower_ch = pd.concat([pd.Series([np.nan] * padding), lower_ch]).values
                    center_ch = pd.concat([pd.Series([np.nan] * padding), center_ch]).values
                else:
                    upper_ch = upper_ch.values[-len(result.upper_channel):]
                    lower_ch = lower_ch.values[-len(result.upper_channel):]
                    center_ch = center_ch.values[-len(result.upper_channel):]
        else:
            upper_ch = result.upper_channel
            lower_ch = result.lower_channel
            center_ch = result.centerline
        
        # プロット作成
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                            gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # メインチャート（ローソク足 + チャネル）
        dates = data.index
        
        # ローソク足（簡易版）
        colors = ['green' if c >= o else 'red' for o, c in zip(data['open'], data['close'])]
        ax1.plot(dates, data['close'], color='blue', linewidth=1, alpha=0.7, label='Close Price')
        
        # Supreme Breakout Channel
        ax1.plot(dates, upper_ch, color='green', linewidth=1.5, label='SBC Upper', alpha=0.8)
        ax1.plot(dates, lower_ch, color='red', linewidth=1.5, label='SBC Lower', alpha=0.8)
        ax1.plot(dates, center_ch, color='navy', linewidth=2, label='SBC Center')
        
        # チャネル塗りつぶし
        ax1.fill_between(dates, upper_ch, lower_ch, alpha=0.1, color='gray')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # トレンド強度
        ax2.plot(dates, result.trend_strength, color='orange', linewidth=1.5, label='Trend Strength')
        ax2.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Trend Strength', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # ヒルベルトトレンド
        ax3.plot(dates, result.hilbert_trend, color='purple', linewidth=1.5, label='Hilbert Trend')
        ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
        ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Hilbert Trend', fontweight='bold')
        ax3.set_xlabel('Date', fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # レポート表示
        report = sbc.get_supreme_intelligence_report()
        report_text = (
            f"Supreme Intelligence: {report.get('supreme_intelligence_score', 0):.3f}\n"
            f"Trend Phase: {report.get('current_trend_phase', 'N/A')}\n"
            f"Signal State: {report.get('current_signal_state', 'N/A')}\n"
            f"Total Signals: {report.get('total_breakout_signals', 0)}"
        )
        
        ax1.text(0.02, 0.98, report_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, result
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン関数"""
    print("🚀 Supreme Breakout Channel Simple Demo 開始!")
    
    # コマンドライン引数処理
    import argparse
    parser = argparse.ArgumentParser(description='Supreme Breakout Channel Simple Demo')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--dummy', action='store_true', help='ダミーデータを使用')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--days', type=int, default=180, help='ダミーデータの日数')
    args = parser.parse_args()
    
    try:
        if args.dummy or not args.config:
            print("📊 ダミーデータを生成中...")
            data = create_dummy_market_data(days=args.days)
            title = f"Supreme Breakout Channel Demo (Dummy Data - {args.days} days)"
        else:
            print("📊 実データを読み込み中...")
            # 設定ファイルからデータを読み込み
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            binance_config = config.get('binance_data', {})
            data_dir = binance_config.get('data_dir', 'data/binance')
            
            binance_data_source = BinanceDataSource(data_dir)
            dummy_csv_source = CSVDataSource("dummy")
            data_loader = DataLoader(
                data_source=dummy_csv_source,
                binance_data_source=binance_data_source
            )
            data_processor = DataProcessor()
            
            raw_data = data_loader.load_data_from_config(config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            first_symbol = next(iter(processed_data))
            data = processed_data[first_symbol]
            title = f"Supreme Breakout Channel - {first_symbol}"
            
            # 最新の1000件に絞る
            if len(data) > 1000:
                data = data.tail(1000)
        
        print(f"✅ データ準備完了 - データ数: {len(data)}")
        print(f"📅 期間: {data.index.min()} → {data.index.max()}")
        
        # チャートプロット
        result = plot_supreme_breakout_channel(data, title)
        
        if result is not None:
            fig, sbc_result = result
            
            # 保存または表示
            if args.output:
                fig.savefig(args.output, dpi=300, bbox_inches='tight')
                print(f"💾 チャートを保存しました: {args.output}")
            else:
                plt.show()
            
            print("\n" + "="*60)
            print("🚀 SUPREME BREAKOUT CHANNEL DEMO 完了! 🚀")
            print("="*60)
        
    except Exception as e:
        print(f"❌ Demo実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 