#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml
from pathlib import Path
import argparse

# インポートパスの設定
# プロジェクトのルートディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# ルートディレクトリをパスに追加
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# データ取得用のクラスをインポート
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーターをインポート
from indicators.alpha_band import AlphaBand
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


def load_sample_data():
    """サンプルOHLCデータをロードする（ここではランダムデータを生成）"""
    np.random.seed(42)  # 再現性のため
    n = 300  # データポイント数
    
    # 基本的な価格トレンドを生成（ランダムウォーク + トレンド）
    base = np.cumsum(np.random.normal(0, 1, n)) + np.linspace(0, 15, n)
    
    # 周期的な成分を追加（より顕著な変動を持たせる）
    cycles = 8 * np.sin(np.linspace(0, 6 * np.pi, n)) + 3 * np.sin(np.linspace(0, 15 * np.pi, n))
    
    # ボラティリティの変化を追加（より変動を大きくする）
    volatility = np.abs(np.sin(np.linspace(0, 4 * np.pi, n))) * 3 + 2
    
    # 終値を作成
    close = base + cycles
    
    # 高値、安値、始値を作成（ボラティリティを強調）
    high = close + np.random.normal(0, 1.5, n) * volatility
    low = close - np.random.normal(0, 1.5, n) * volatility
    open_ = close - np.random.normal(0, 0.8, n) * volatility
    
    # DataFrameに変換
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return df


def print_statistics(name, values):
    """配列の統計情報を表示する"""
    values_clean = values[~np.isnan(values)]  # NaN値を除外
    if len(values_clean) == 0:
        print(f"{name}: すべての値がNaNです")
        return
    
    print(f"{name}の統計情報:")
    print(f"  平均値: {np.mean(values_clean):.6f}")
    print(f"  最小値: {np.min(values_clean):.6f}")
    print(f"  最大値: {np.max(values_clean):.6f}")
    print(f"  標準偏差: {np.std(values_clean):.6f}")
    print(f"  NaNの数: {np.sum(np.isnan(values))}")
    print(f"  ゼロの数: {np.sum(values_clean == 0)}")
    print(f"  要素数: {len(values)}")


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='AlphaBandの計算と表示')
    parser.add_argument('--symbol', '-s', type=str, help='使用する通貨ペア（例: BTC_future）')
    parser.add_argument('--cycle-detector', '-c', type=str, default='hody_dc', 
                       choices=['dudi_dc', 'hody_dc', 'phac_dc', 'dudi_dce', 'hody_dce', 'phac_dce'],
                       help='サイクル検出器の種類')
    parser.add_argument('--lp-period', type=int, default=5, help='ローパスフィルターの期間')
    parser.add_argument('--hp-period', type=int, default=144, help='ハイパスフィルターの期間')
    parser.add_argument('--max-mult', type=float, default=3.0, help='ATR乗数の最大値')
    parser.add_argument('--min-mult', type=float, default=1.5, help='ATR乗数の最小値')
    parser.add_argument('--limit', type=int, default=None, help='使用するデータポイント数の制限')
    parser.add_argument('--sample', action='store_true', help='サンプルデータを使用する')
    parser.add_argument('--smoother', type=str, default='hyper', choices=['alma', 'hyper'],
                       help='平滑化アルゴリズム（alma: ALMA, hyper: ハイパースムーサー）')
    return parser.parse_args()


def main():
    """メイン関数"""
    # コマンドライン引数をパース
    args = parse_args()
    
    # サンプルデータを使うかどうか
    use_sample_data = args.sample
    target_symbol = args.symbol
    
    if not use_sample_data:
        # 設定ファイルの読み込み
        config_path = Path(root_dir) / 'config.yaml'
        
        # configファイルが存在するか確認
        if not config_path.exists():
            print(f"設定ファイルが見つかりません: {config_path}")
            print("サンプルデータを使用します...")
            df = load_sample_data()
            use_sample_data = True
            symbol = "sample"
        else:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # データの準備
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
                print("\nデータの読み込みと処理中...")
                raw_data = data_loader.load_data_from_config(config)
                processed_data = {
                    sym: data_processor.process(df)
                    for sym, df in raw_data.items()
                }
                
                if not processed_data:
                    print("データが読み込めませんでした。サンプルデータを使用します...")
                    df = load_sample_data()
                    use_sample_data = True
                    symbol = "sample"
                else:
                    # 引数で指定されたシンボルを使用するか、指定がなければ最初のシンボル
                    if target_symbol and target_symbol in processed_data:
                        symbol = target_symbol
                    else:
                        available_symbols = list(processed_data.keys())
                        if target_symbol:
                            print(f"指定されたシンボル '{target_symbol}' が見つかりません。")
                            print(f"利用可能なシンボル: {available_symbols}")
                        
                        symbol = next(iter(processed_data))
                        print(f"シンボル '{symbol}' を使用します。")
                    
                    df = processed_data[symbol]
                    
                    # データポイント数の制限があれば適用
                    if args.limit is not None and args.limit > 0:
                        if args.limit < len(df):
                            df = df.iloc[-args.limit:]
                            print(f"データを最新の {args.limit} ポイントに制限しました。")
                    
                    # データの状態を確認
                    print(f"データの形状: {df.shape}")
                    print(f"カラム: {df.columns.tolist()}")
                    print(f"期間: {df.index[0]} から {df.index[-1]}")
                    print(f"NaNの数: {df.isna().sum().sum()}")
            except Exception as e:
                print(f"データロード中にエラーが発生しました: {e}")
                print("サンプルデータを使用します...")
                df = load_sample_data()
                use_sample_data = True
                symbol = "sample"
    else:
        # サンプルデータを使用
        print("サンプルデータを使用します...")
        df = load_sample_data()
        symbol = "sample"
    
    # AlphaBandの計算
    print("\nAlphaBandを計算しています...")
    print(f"使用パラメータ: サイクル検出器={args.cycle_detector}, LP期間={args.lp_period}, HP期間={args.hp_period}")
    print(f"ATR乗数: 最大={args.max_mult}, 最小={args.min_mult}")
    print(f"平滑化アルゴリズム: {args.smoother}")
    
    alpha_band = AlphaBand(
        cycle_detector_type=args.cycle_detector,
        lp_period=args.lp_period,
        hp_period=args.hp_period,
        cycle_part=0.5,
        max_kama_period=55,
        min_kama_period=8,
        max_atr_period=55, 
        min_atr_period=8,
        max_multiplier=args.max_mult,
        min_multiplier=args.min_mult,
        smoother_type=args.smoother
    )
    
    middle = alpha_band.calculate(df)
    middle, upper, lower = alpha_band.get_bands()
    cer = alpha_band.get_cycle_er()
    dynamic_multiplier = alpha_band.get_dynamic_multiplier()
    alpha_atr_values = alpha_band.get_alpha_atr()
    
    # サイクル効率比の追加計算（比較のため）
    print("\nサイクル効率比を計算しています...")
    cycle_er = CycleEfficiencyRatio(
        cycle_detector_type=args.cycle_detector,
        lp_period=args.lp_period,
        hp_period=args.hp_period,
        cycle_part=0.5
    )
    cer_standalone = cycle_er.calculate(df)
    cycles = cycle_er.get_cycles()
    
    # 追加の診断情報を取得
    alpha_atr_percent = alpha_band.alpha_atr.get_percent_atr()  # パーセントベースのATR
    alpha_atr_absolute = alpha_band.alpha_atr.get_absolute_atr()  # 金額ベースのATR
    tr_values = alpha_band.alpha_atr.get_true_range()  # True Range値
    
    # ========== 統計情報の出力 ==========
    print("\n============ 統計情報 ============")
    print_statistics("中心線 (Middle)", middle)
    print_statistics("上限バンド (Upper)", upper)
    print_statistics("下限バンド (Lower)", lower)
    print_statistics("サイクル効率比 (CER)", cer)
    print_statistics("動的乗数 (Dynamic Multiplier)", dynamic_multiplier)
    print_statistics("AlphaATR値 (絶対値)", alpha_atr_absolute)
    print_statistics("AlphaATR値 (パーセント)", alpha_atr_percent)
    print_statistics("True Range", tr_values)
    
    # バンド間の差異を計算
    upper_width = upper - middle
    lower_width = middle - lower
    
    print("\n============ バンド幅の情報 ============")
    print_statistics("上側バンド幅 (Upper - Middle)", upper_width)
    print_statistics("下側バンド幅 (Middle - Lower)", lower_width)
    
    # サイクル効率比とバンド幅の相関関係を計算
    valid_indices = ~np.isnan(cer) & ~np.isnan(upper_width)
    if np.sum(valid_indices) > 2:
        cer_width_corr = np.corrcoef(cer[valid_indices], upper_width[valid_indices])[0, 1]
        print(f"\nCERとバンド幅の相関係数: {cer_width_corr:.6f}")
    
    # 問題検出のためのチェック
    print("\n============ 問題検出 ============")
    # バンド幅がほぼゼロの場所を検出
    zero_band_width = np.sum((upper_width < 1e-6) & (upper_width > 0))
    print(f"上側バンド幅がほぼゼロ (< 1e-6) の数: {zero_band_width}")
    
    zero_dynamic_mult = np.sum((dynamic_multiplier < 1e-6) & (dynamic_multiplier > 0))
    print(f"動的乗数がほぼゼロ (< 1e-6) の数: {zero_dynamic_mult}")
    
    zero_alpha_atr = np.sum((alpha_atr_absolute < 1e-6) & (alpha_atr_absolute > 0))
    print(f"AlphaATR値（絶対値）がほぼゼロ (< 1e-6) の数: {zero_alpha_atr}")
    
    print("\n============ データバンド比率 ============")
    # 平均値の統計情報
    close_values = df['close'].values
    close_mean = np.mean(close_values[~np.isnan(close_values)])
    atr_abs_mean = np.mean(alpha_atr_absolute[~np.isnan(alpha_atr_absolute)])
    # バンド幅と価格の比率
    band_width_ratio = atr_abs_mean / close_mean
    print(f"平均ATR(絶対値) / 平均価格: {band_width_ratio:.6f}")
    print(f"ATR倍率適用後の期待バンド幅: {band_width_ratio * np.mean(dynamic_multiplier[~np.isnan(dynamic_multiplier)]):.6f}")
    
    # プロット
    print("\n結果をプロットしています...")
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.2)
    
    # 使用データの情報をタイトルに追加
    title_prefix = f"{symbol} - "
    param_text = f"[{args.cycle_detector}, LP={args.lp_period}, HP={args.hp_period}, Mult={args.min_mult}-{args.max_mult}]"
    
    # 価格とバンドのプロット
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Close', color='black', alpha=0.7)
    ax1.plot(df.index, middle, label='AlphaBand Middle', color='blue', linewidth=1.5)
    ax1.plot(df.index, upper, label='AlphaBand Upper', color='green', linewidth=1.2, linestyle='--')
    ax1.plot(df.index, lower, label='AlphaBand Lower', color='red', linewidth=1.2, linestyle='--')
    ax1.set_title(f'{title_prefix}AlphaBand {param_text}')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 塗りつぶしでバンド幅を視覚化
    ax1.fill_between(df.index, upper, middle, color='green', alpha=0.1)
    ax1.fill_between(df.index, middle, lower, color='red', alpha=0.1)
    
    # バンド幅のプロット
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    band_width = upper - lower
    ax2.plot(df.index, band_width, label='バンド幅 (Upper - Lower)', color='purple', linewidth=1.2)
    ax2.set_ylabel('Band Width')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # サイクル効率比のプロット
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, cer, label='Cycle Efficiency Ratio', color='blue', linewidth=1.2)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('CER Value')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 動的乗数のプロット
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, dynamic_multiplier, label='Dynamic Multiplier', color='orange', linewidth=1.2)
    ax4.set_ylabel('Multiplier')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # ATR値とTRのプロット
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(df.index, alpha_atr_absolute, label='AlphaATR (絶対値)', color='magenta', linewidth=1.2)
    ax5.plot(df.index, tr_values, label='True Range', color='gray', linewidth=1.0, alpha=0.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('ATR/TR Value')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # グラフを保存
    try:
        output_dir = Path(current_dir) / 'output'
        output_dir.mkdir(exist_ok=True)
        param_str = f"{args.cycle_detector}_lp{args.lp_period}_hp{args.hp_period}_mult{args.min_mult}-{args.max_mult}"
        file_name = f'alpha_band_{symbol}_{param_str}.png'
        output_file = output_dir / file_name
        plt.savefig(output_file)
        print(f"\nグラフを保存しました: {output_file}")
    except Exception as e:
        print(f"グラフの保存中にエラーが発生しました: {e}")
    
    plt.show()
    print("\n完了しました。")


if __name__ == "__main__":
    main() 