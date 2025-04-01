#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# カレントディレクトリをプロジェクトのルートに設定
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# インポート
from indicators import AlphaVIX, AlphaVolatility, AlphaATR, AlphaER
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource


def generate_sample_data(n=500):
    """
    テスト用のOHLCデータを生成する
    """
    # 初期価格
    base_price = 70000
    # 初期変動サイズ
    volatility = 0.01
    
    # ランダムな市場状態を定義（トレンド/レンジ）
    market_states = []
    current_state = "trend"  # 最初はトレンド
    state_length = random.randint(30, 60)  # 最初の状態の長さ
    
    for i in range(n):
        if i % state_length == 0:
            # 状態を変更する確率
            if current_state == "trend":
                current_state = "range" if random.random() < 0.7 else "trend"
            else:
                current_state = "trend" if random.random() < 0.7 else "range"
            state_length = random.randint(30, 60)  # 新しい状態の長さ
        
        market_states.append(current_state)
    
    # ボラティリティ状態の定義
    volatility_states = []
    current_vol = "normal"  # 最初は通常ボラティリティ
    vol_length = random.randint(40, 80)
    
    for i in range(n):
        if i % vol_length == 0:
            # ボラティリティ状態の変更
            r = random.random()
            if r < 0.2:
                current_vol = "high"
            elif r < 0.4:
                current_vol = "low"
            else:
                current_vol = "normal"
            vol_length = random.randint(40, 80)
        
        volatility_states.append(current_vol)
    
    # 価格データの生成
    close = [base_price]
    high = [base_price * (1 + volatility/2)]
    low = [base_price * (1 - volatility/2)]
    
    for i in range(1, n):
        # 現在の状態に基づいてトレンドを決定
        if market_states[i] == "trend":
            trend = random.choice([-1, 1]) * 0.003  # トレンド方向
        else:
            trend = 0  # レンジ相場
        
        # 現在のボラティリティ状態に基づいて変動サイズを決定
        if volatility_states[i] == "high":
            vol_factor = random.uniform(1.5, 3.0)
        elif volatility_states[i] == "low":
            vol_factor = random.uniform(0.3, 0.7)
        else:
            vol_factor = random.uniform(0.8, 1.2)
        
        current_volatility = volatility * vol_factor
        
        # 前日の終値からの変動
        price_change = close[-1] * (trend + random.gauss(0, current_volatility))
        new_close = close[-1] + price_change
        
        # 急激な価格変動を追加（10%の確率）
        if random.random() < 0.1:
            spike_factor = random.choice([-1, 1]) * random.uniform(0.01, 0.03)
            new_close *= (1 + spike_factor)
        
        # 高値と安値の計算（現実的な範囲内）
        daily_range = new_close * current_volatility * random.uniform(0.8, 1.2)
        new_high = new_close + daily_range * random.uniform(0.4, 0.6)
        new_low = new_close - daily_range * random.uniform(0.4, 0.6)
        
        # 時系列の最小値は安値、最大値は高値となるよう調整
        if new_low > min(close):
            new_low = min(close) * random.uniform(0.995, 0.999)
        
        if new_high < max(close):
            new_high = max(close) * random.uniform(1.001, 1.005)
        
        close.append(new_close)
        high.append(new_high)
        low.append(new_low)
    
    # DataFrameの作成 - open列は前日の終値を使用
    open_prices = close[:-1]  # 最後の終値は次の日の始値にならないため、1つ少ない
    
    # すべての配列の長さを一致させる
    n_actual = min(len(open_prices), len(high), len(low), len(close), len(market_states), len(volatility_states))
    
    df = pd.DataFrame({
        'open': open_prices[:n_actual],
        'high': high[:n_actual],
        'low': low[:n_actual],
        'close': close[:n_actual],
        'market_state': market_states[:n_actual],
        'volatility_state': volatility_states[:n_actual]
    })
    
    return df


def load_binance_data(symbol, timeframe, start_date, end_date, data_dir='data/binance'):
    """
    Binanceからデータをロードする
    
    Args:
        symbol: 通貨ペア（例: 'BTCUSDT'）
        timeframe: 時間枠（例: '1d', '4h', '1h'）
        start_date: 開始日時（文字列 'YYYY-MM-DD' または datetime）
        end_date: 終了日時（文字列 'YYYY-MM-DD' または datetime）
        data_dir: Binanceデータのディレクトリ
        
    Returns:
        pandas.DataFrame: ローソク足データ
    """
    print(f"Binanceデータのロード: {symbol} {timeframe} {start_date} → {end_date}")
    
    try:
        # 日付の処理
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # ディレクトリの存在確認
        import os
        symbol_dir = os.path.join(data_dir, symbol)
        
        # 利用可能な時間枠を確認
        available_timeframes = []
        if os.path.exists(symbol_dir):
            for item in os.listdir(symbol_dir):
                item_path = os.path.join(symbol_dir, item)
                if os.path.isdir(item_path):
                    available_timeframes.append(item)
        
        # 指定された時間枠が利用可能でない場合は代替を探す
        if timeframe not in available_timeframes:
            if "4h" in available_timeframes:
                print(f"指定された時間枠 {timeframe} は見つかりませんでした。4h を使用します。")
                timeframe = "4h"
            elif available_timeframes:
                first_available = available_timeframes[0]
                print(f"指定された時間枠 {timeframe} は見つかりませんでした。{first_available} を使用します。")
                timeframe = first_available
            else:
                print(f"シンボル {symbol} に利用可能な時間枠が見つかりません。")
                return None
        
        # BinanceDataSourceの初期化
        binance_data_source = BinanceDataSource(data_dir)
        
        # t.pyと同様の方法でデータを直接読み込む
        try:
            # binance_data_source.load_dataを直接使用
            df = binance_data_source.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"データ読み込み完了: {len(df)} レコード")
            return df
            
        except Exception as e:
            print(f"BinanceDataSourceからの直接読み込みエラー: {e}")
            
            # CSVデータソースとDataLoaderを使用した代替方法
            dummy_csv_source = CSVDataSource("dummy")
            data_loader = DataLoader(
                data_source=dummy_csv_source,
                binance_data_source=binance_data_source
            )
            
            # config辞書を作成してload_data_from_configを使用
            config = {
                'binance_data': {
                    'enabled': True,
                    'symbols': [symbol],
                    'timeframes': [timeframe],
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
            
            try:
                data_dict = data_loader.load_data_from_config(config)
                if data_dict and symbol in data_dict:
                    df = data_dict[symbol]
                    print(f"config経由でデータ読み込み完了: {len(df)} レコード")
                    return df
                else:
                    raise ValueError(f"データが見つかりません: {symbol}")
            except Exception as e2:
                print(f"config経由の読み込みエラー: {e2}")
                return None
            
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
    
    print("データの読み込みに失敗しました。")
    return None


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='Alpha VIXインジケーターの実データテスト')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='テストする通貨ペア (デフォルト: BTCUSDT)')
    
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='時間枠 (デフォルト: 1d, 他: 4h, 1h など)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='開始日 (YYYY-MM-DD形式, デフォルト: 1年前)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='終了日 (YYYY-MM-DD形式, デフォルト: 今日)')
    
    parser.add_argument('--data-dir', type=str, default='data/binance',
                        help='Binanceデータのディレクトリ (デフォルト: data/binance)')
    
    parser.add_argument('--use-sample-data', action='store_true',
                        help='Binanceデータの代わりにサンプルデータを使用')
    
    parser.add_argument('--sample-size', type=int, default=500,
                        help='サンプルデータのサイズ (デフォルト: 500)')
    
    parser.add_argument('--er-period', type=int, default=21,
                        help='効率比の計算期間 (デフォルト: 21)')
    
    parser.add_argument('--max-period', type=int, default=89,
                        help='VIX計算用の最大期間 (デフォルト: 89)')
    
    parser.add_argument('--min-period', type=int, default=13,
                        help='VIX計算用の最小期間 (デフォルト: 13)')
    
    parser.add_argument('--smoothing-period', type=int, default=14,
                        help='平滑化期間 (デフォルト: 14)')
    
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple', 'weighted', 'maximum', 'harmonic', 'logarithmic'],
                        help='計算モード (デフォルト: simple)')
    
    parser.add_argument('--use-config', action='store_true',
                        help='config.yamlからデータを読み込む')
    
    parser.add_argument('--config-path', type=str, default='config.yaml',
                        help='設定ファイルへのパス (デフォルト: config.yaml)')
    
    return parser.parse_args()


def load_data_from_config(config_path='config.yaml'):
    """
    config.yamlからデータ設定を読み込み、データを取得する
    
    Args:
        config_path: 設定ファイルへのパス
        
    Returns:
        DataFrame: 取得されたデータ
    """
    print(f"設定ファイル {config_path} からデータ設定を読み込みます...")
    
    try:
        # 設定ファイルの読み込み
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Binanceデータ設定を取得
        binance_config = config.get('binance_data', {})
        
        if not binance_config.get('enabled', False):
            print("Binanceデータは無効です。コマンドライン引数またはデフォルト設定を使用します。")
            return None
        
        # データソース設定
        data_dir = binance_config.get('data_dir', 'data/binance')
        symbol = binance_config.get('symbol', 'BTC')
        timeframe = binance_config.get('timeframe', '4h')
        market_type = binance_config.get('market_type', None)
        start_date = binance_config.get('start', '2023-01-01')
        end_date = binance_config.get('end', datetime.now().strftime('%Y-%m-%d'))
        
        # 必要なクラスを初期化
        binance_data_source = BinanceDataSource(data_dir)
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        
        print(f"\n設定ファイルから読み込まれた設定:")
        print(f"シンボル: {symbol}, 時間枠: {timeframe}, 市場タイプ: {market_type}")
        print(f"期間: {start_date} → {end_date}")
        print(f"データディレクトリ: {data_dir}")
        
        # データの読み込み
        try:
            if market_type:
                # 市場タイプが指定されている場合
                df = binance_data_source.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    market_type=market_type
                )
            else:
                # 市場タイプが指定されていない場合
                df = binance_data_source.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if df is not None and not df.empty:
                print(f"データ読み込み完了: {len(df)} レコード")
                return df
            else:
                # config全体を使ってデータを読み込む
                raw_data = data_loader.load_data_from_config(config)
                
                if raw_data and symbol in raw_data:
                    df = raw_data[symbol]
                    print(f"config経由でデータ読み込み完了: {len(df)} レコード")
                    return df
                elif raw_data:
                    # 最初に見つかったデータを使用
                    first_key = next(iter(raw_data))
                    df = raw_data[first_key]
                    print(f"シンボル {symbol} のデータが見つかりませんでした。{first_key} を使用します。")
                    print(f"config経由でデータ読み込み完了: {len(df)} レコード")
                    return df
        except Exception as e:
            print(f"設定ファイルからのデータ読み込みエラー: {e}")
    
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
    
    print("設定ファイルからのデータ読み込みに失敗しました。")
    return None


def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    args = parse_args()
    
    print("アルファVIXテスト\n")
    
    # データの準備
    if args.use_sample_data:
        print(f"サンプルデータを生成: {args.sample_size} レコード")
        df = generate_sample_data(n=args.sample_size)
    elif args.use_config:
        # 設定ファイルからデータを読み込む
        df = load_data_from_config(args.config_path)
        
        if df is None or df.empty:
            print("設定ファイルからのデータ読み込みに失敗しました。サンプルデータを使用します。")
            df = generate_sample_data(n=args.sample_size)
    else:
        # 日付範囲の設定
        if args.start_date is None:
            # デフォルトは1年前
            start_date = datetime.now() - timedelta(days=365)
            start_date = start_date.strftime('%Y-%m-%d')
        else:
            start_date = args.start_date
            
        if args.end_date is None:
            # デフォルトは今日
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = args.end_date
        
        # Binanceデータのロード
        df = load_binance_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            data_dir=args.data_dir
        )
        
        if df is None or df.empty:
            print("データが取得できませんでした。サンプルデータを使用します。")
            df = generate_sample_data(n=args.sample_size)
    
    # パラメータの設定
    er_period = args.er_period
    max_period = args.max_period
    min_period = args.min_period
    smoothing_period = args.smoothing_period
    mode = args.mode
    
    print(f"\nパラメータ設定:")
    print(f"ER期間: {er_period}, 最大期間: {max_period}, 最小期間: {min_period}")
    print(f"平滑化期間: {smoothing_period}, モード: {mode}")
    
    # Alpha VIXの計算
    alpha_vix = AlphaVIX(
        er_period=er_period, 
        max_period=max_period,
        min_period=min_period,
        smoothing_period=smoothing_period,
        mode=mode
    )
    vix_values = alpha_vix.calculate(df)
    
    # サブインジケーターの取得
    er_values = alpha_vix.get_efficiency_ratio()
    atr_values = alpha_vix.get_percent_atr()
    vol_values = alpha_vix.get_percent_volatility()
    
    # 金額ベースのVIX値を取得
    absolute_vix = alpha_vix.get_absolute_vix()
    
    # 追加のインディケーター
    alpha_er = AlphaER(period=er_period)
    er_values_alpha = alpha_er.calculate(df)
    
    # 結果をDataFrameに追加
    df['vix'] = vix_values
    df['absolute_vix'] = absolute_vix
    df['er'] = er_values
    df['atr'] = atr_values
    df['vol'] = vol_values
    
    # 統計情報
    print(f"------- Alpha VIX 統計 -------")
    print(f"VIX % 平均: {df['vix'].mean():.4f}%")
    print(f"VIX % 標準偏差: {df['vix'].std():.4f}%")
    print(f"VIX % 最小: {df['vix'].min():.4f}%")
    print(f"VIX % 最大: {df['vix'].max():.4f}%")
    print()
    
    print(f"VIX 金額ベース平均: {df['absolute_vix'].mean():.4f}")
    print(f"VIX 金額ベース最大: {df['absolute_vix'].max():.4f}")
    print()
    
    print(f"ATR % 平均: {df['atr'].mean():.4f}%")
    print(f"Volatility % 平均: {df['vol'].mean():.4f}%")
    print()
    
    # 相関行列
    corr_columns = ['vix', 'absolute_vix', 'er', 'atr', 'vol']
    correlation = df[corr_columns].corr()
    
    print("------- 相関行列 -------")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(correlation)
    print()
    
    # 効率比（ER）とATRの関係
    er_atr_corr = np.corrcoef(df['er'].dropna(), df['atr'].dropna())[0, 1]
    print(f"効率比とATRの相関: {er_atr_corr:.6f}")
    
    # 効率比（ER）とボラティリティの関係
    er_vol_corr = np.corrcoef(df['er'].dropna(), df['vol'].dropna())[0, 1]
    print(f"効率比とボラティリティの相関: {er_vol_corr:.6f}")
    
    # トレンド中と範囲相場のボラティリティ分析
    strong_trend_df = df[df['er'] > 0.6]
    range_df = df[df['er'] < 0.4]
    
    if not strong_trend_df.empty:
        print(f"強いトレンド時の平均VIX (ER > 0.6): {strong_trend_df['vix'].mean():.4f}%")
    
    if not range_df.empty:
        print(f"範囲相場時の平均VIX (ER < 0.4): {range_df['vix'].mean():.4f}%")
    
    print()
    
    # ポジションサイジングの例
    capital = 10000.00  # 資金
    risk_percent = 0.01  # リスク率（資金の1%）
    risk_amount = capital * risk_percent
    
    # 最新の価格とボラティリティ
    current_price = df['close'].iloc[-1]
    current_vix_percent = df['vix'].iloc[-1]
    current_vix_absolute = df['absolute_vix'].iloc[-1]
    
    # ボラティリティに基づくポジションサイズの計算
    print("------- ポジションサイジング例 -------")
    print(f"資本: {capital:.2f}")
    print(f"リスク金額 (資本の{risk_percent*100}%): {risk_amount:.2f}")
    print(f"現在価格: {current_price:.2f}")
    print(f"現在 % VIX: {current_vix_percent:.4f}%")
    print(f"現在金額ベースVIX: {current_vix_absolute:.4f}")
    
    # 異なるボラティリティ倍数でのリスク計算
    print("\n異なるボラティリティ倍数でのリスク:")
    for mult in [0.5, 1.0, 2.0, 3.0]:
        vix_risk = current_vix_absolute * mult
        risk_pct = vix_risk / current_price
        print(f"{mult}x VIX: {vix_risk:.2f} ({risk_pct*100:.4f}%)")
    
    # 1x VIXでのポジションサイズ
    position_size = risk_amount / current_vix_absolute  # 1倍のボラティリティリスク
    position_value = position_size * current_price
    
    print(f"\nポジションサイズ (1x VIXでリスク): {position_size:.4f}")
    print(f"ポジション価値: {position_value:.2f}")
    print(f"このポジションは約 {risk_amount / (position_size * current_vix_absolute):.2f}x の現在のボラティリティをリスクにしています")
    
    # 別途AlphaATRとAlphaVolatilityも計算してみる
    alpha_atr = AlphaATR(
        er_period=er_period,
        max_atr_period=max_period,
        min_atr_period=min_period
    )
    
    alpha_volatility = AlphaVolatility(
        er_period=er_period,
        max_vol_period=max_period,
        min_vol_period=min_period,
        smoothing_period=smoothing_period
    )
    
    # 計算
    alpha_atr.calculate(df)
    alpha_volatility.calculate(df)
    
    # 金額ベースと%ベースの値を取得
    atr_percent = alpha_atr.get_percent_atr()
    atr_absolute = alpha_atr.get_absolute_atr()
    
    vol_percent = alpha_volatility.get_percent_volatility()
    vol_absolute = alpha_volatility.get_absolute_volatility()
    
    # DataFrameに追加
    df['atr_pct'] = atr_percent
    df['atr_abs'] = atr_absolute
    df['vol_pct'] = vol_percent
    df['vol_abs'] = vol_absolute
    
    # 統計情報の追加
    print("------- ATRとVolatilityの比較 -------")
    print(f"ATR % 平均: {df['atr_pct'].mean():.4f}%")
    print(f"Volatility % 平均: {df['vol_pct'].mean():.4f}%")
    print(f"VIX % 平均: {df['vix'].mean():.4f}%")
    print()
    
    print(f"ATR 金額 平均: {df['atr_abs'].mean():.4f}")
    print(f"Volatility 金額 平均: {df['vol_abs'].mean():.4f}")
    print(f"VIX 金額 平均: {df['absolute_vix'].mean():.4f}")
    print()
    
    # ===== グラフ表示 =====
    # 元のグラフ表示コードをコメントアウトまたは削除
    
    # 新しいグラフセットの作成
    fig = plt.figure(figsize=(18, 12))
    
    # 6つのサブプロットを作成（6x1のグリッド）
    gs = GridSpec(6, 1, height_ratios=[2, 1, 1, 1, 1, 1])
    
    # データの期間を表示タイトルに追加
    if args.use_config:
        # config.yamlから読み込んだ場合
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        binance_config = config.get('binance_data', {})
        symbol = binance_config.get('symbol', 'BTC')
        timeframe = binance_config.get('timeframe', '4h')
        if binance_config.get('market_type'):
            symbol = f"{symbol} ({binance_config.get('market_type')})"
    else:
        symbol = args.symbol if not args.use_sample_data else "Sample Data"
        timeframe = args.timeframe if not args.use_sample_data else ""
    
    start_date_str = df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else "Start"
    end_date_str = df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else "End"
    chart_title = f"Alpha Indicators Comparison: {symbol} {timeframe} ({start_date_str} to {end_date_str})"
    
    # 価格チャート（一番上）
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Close', color='#1f77b4')
    ax1.set_ylabel('Price')
    ax1.set_title(chart_title)
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # アルファVIX（2番目）
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['vix'], 'r-', label='Alpha VIX (%)', linewidth=1.5)
    ax2.set_ylabel('VIX (%)')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # アルファATR（3番目）
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['atr_pct'], 'm-', label='Alpha ATR (%)', linewidth=1.5)
    ax3.set_ylabel('ATR (%)')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    
    # アルファボラティリティ（4番目）
    ax4 = plt.subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, df['vol_pct'], 'c-', label='Alpha Volatility (%)', linewidth=1.5)
    ax4.set_ylabel('Vol (%)')
    ax4.grid(True)
    ax4.legend(loc='upper left')
    
    # 効率比（5番目）
    ax5 = plt.subplot(gs[4], sharex=ax1)
    ax5.plot(df.index, df['er'], 'g-', label='Efficiency Ratio', linewidth=1.5)
    ax5.set_ylabel('ER')
    ax5.set_ylim(-0.1, 1.1)
    ax5.grid(True)
    ax5.legend(loc='upper left')
    
    # すべてのインジケーターを一緒に表示（6番目）
    ax6 = plt.subplot(gs[5], sharex=ax1)
    ax6.plot(df.index, df['vix'], 'r-', label='VIX (%)', alpha=0.8)
    ax6.plot(df.index, df['atr_pct'], 'm-', label='ATR (%)', alpha=0.8)
    ax6.plot(df.index, df['vol_pct'], 'c-', label='Volatility (%)', alpha=0.8)
    ax6.set_ylabel('Indicators (%)')
    ax6.set_xlabel('Time')
    ax6.grid(True)
    ax6.legend(loc='upper left')
    
    # x軸の日付フォーマットを設定
    if not args.use_sample_data:
        from matplotlib.dates import DateFormatter
        date_format = DateFormatter('%Y-%m-%d')
        ax6.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 詳細な比較分析
    print("\n------- インジケーター詳細比較分析 -------")
    
    # 各インジケーターの基本統計量
    print("基本統計量:")
    indicators_stats = df[['vix', 'atr_pct', 'vol_pct']].describe()
    indicators_stats = indicators_stats.rename(columns={
        'vix': 'Alpha VIX (%)',
        'atr_pct': 'Alpha ATR (%)',
        'vol_pct': 'Alpha Volatility (%)'
    })
    print(indicators_stats)
    
    # 相関行列の詳細表示
    print("\n------- 詳細相関行列 -------")
    detailed_correlation = df[['vix', 'atr_pct', 'vol_pct', 'er', 'absolute_vix', 'atr_abs', 'vol_abs']].corr()
    detailed_correlation = detailed_correlation.rename(columns={
        'vix': 'Alpha VIX (%)',
        'atr_pct': 'Alpha ATR (%)',
        'vol_pct': 'Alpha Vol (%)',
        'er': 'ER',
        'absolute_vix': 'VIX (Abs)',
        'atr_abs': 'ATR (Abs)',
        'vol_abs': 'Vol (Abs)'
    }, index={
        'vix': 'Alpha VIX (%)',
        'atr_pct': 'Alpha ATR (%)',
        'vol_pct': 'Alpha Vol (%)',
        'er': 'ER',
        'absolute_vix': 'VIX (Abs)',
        'atr_abs': 'ATR (Abs)',
        'vol_abs': 'Vol (Abs)'
    })
    print(detailed_correlation)
    
    # 各指標の寄与度分析
    print("\n------- VIXに対する各指標の寄与度 -------")
    # ATRの寄与度
    atr_contribution = df['atr_pct'] / (df['atr_pct'] + df['vol_pct'])
    atr_contribution = atr_contribution.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"ATRの平均寄与度: {atr_contribution.mean():.4f} ({atr_contribution.mean()*100:.2f}%)")
    print(f"Volatilityの平均寄与度: {1-atr_contribution.mean():.4f} ({(1-atr_contribution.mean())*100:.2f}%)")
    
    # 効率比による条件付き分析
    high_er = df[df['er'] > 0.6]
    low_er = df[df['er'] < 0.4]
    
    print("\n------- 異なる市場状態での比較 -------")
    print(f"サンプル数 - 強トレンド (ER>0.6): {len(high_er)}, 範囲相場 (ER<0.4): {len(low_er)}")
    
    if not high_er.empty:
        print(f"\n強トレンド時 (ER > 0.6):")
        print(f"Alpha VIX: {high_er['vix'].mean():.4f}%")
        print(f"Alpha ATR: {high_er['atr_pct'].mean():.4f}%")
        print(f"Alpha Vol: {high_er['vol_pct'].mean():.4f}%")
        
        # 強トレンド時のATR寄与度
        if 'atr_pct' in high_er.columns and 'vol_pct' in high_er.columns:
            trend_atr_contrib = high_er['atr_pct'] / (high_er['atr_pct'] + high_er['vol_pct'])
            trend_atr_contrib = trend_atr_contrib.replace([np.inf, -np.inf], np.nan).dropna()
            if not trend_atr_contrib.empty:
                print(f"トレンド時ATR寄与度: {trend_atr_contrib.mean():.4f} ({trend_atr_contrib.mean()*100:.2f}%)")
    
    if not low_er.empty:
        print(f"\n範囲相場時 (ER < 0.4):")
        print(f"Alpha VIX: {low_er['vix'].mean():.4f}%")
        print(f"Alpha ATR: {low_er['atr_pct'].mean():.4f}%")
        print(f"Alpha Vol: {low_er['vol_pct'].mean():.4f}%")
        
        # 範囲相場時のATR寄与度
        if 'atr_pct' in low_er.columns and 'vol_pct' in low_er.columns:
            range_atr_contrib = low_er['atr_pct'] / (low_er['atr_pct'] + low_er['vol_pct'])
            range_atr_contrib = range_atr_contrib.replace([np.inf, -np.inf], np.nan).dropna()
            if not range_atr_contrib.empty:
                print(f"範囲相場時ATR寄与度: {range_atr_contrib.mean():.4f} ({range_atr_contrib.mean()*100:.2f}%)")
    
    # ヒストグラム比較
    plt.figure(figsize=(18, 6))
    
    # 3つのサブプロットを横に並べる
    plt.subplot(131)
    plt.hist(df['vix'].dropna(), bins=50, alpha=0.7, color='red')
    plt.title('Alpha VIX (%) Distribution')
    plt.xlabel('VIX (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(132)
    plt.hist(df['atr_pct'].dropna(), bins=50, alpha=0.7, color='magenta')
    plt.title('Alpha ATR (%) Distribution')
    plt.xlabel('ATR (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(133)
    plt.hist(df['vol_pct'].dropna(), bins=50, alpha=0.7, color='cyan')
    plt.title('Alpha Volatility (%) Distribution')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # インジケーター間の散布図
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.scatter(df['atr_pct'], df['vix'], alpha=0.5, color='crimson')
    plt.title('ATR vs VIX')
    plt.xlabel('Alpha ATR (%)')
    plt.ylabel('Alpha VIX (%)')
    plt.grid(True, alpha=0.3)
    
    # 相関係数と傾き（線形回帰）を表示
    if 'atr_pct' in df.columns and 'vix' in df.columns:
        atr_vix_corr = df['atr_pct'].corr(df['vix'])
        x = df['atr_pct'].dropna().values
        y = df['vix'].dropna().values
        if len(x) > 1 and len(y) > 1:
            try:
                from sklearn.linear_model import LinearRegression
                x_reshaped = x.reshape(-1, 1)
                model = LinearRegression().fit(x_reshaped, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                plt.plot(x, intercept + slope * x, 'r--', alpha=0.7)
                plt.text(0.05, 0.95, f'Corr: {atr_vix_corr:.4f}\nSlope: {slope:.4f}', 
                        transform=plt.gca().transAxes, fontsize=9, 
                        verticalalignment='top')
            except:
                # sklearn.linear_model をインポートできない場合や
                # 線形回帰モデルのフィットに失敗した場合は相関係数のみを表示
                plt.text(0.05, 0.95, f'Corr: {atr_vix_corr:.4f}', 
                        transform=plt.gca().transAxes, fontsize=9, 
                        verticalalignment='top')
    
    plt.subplot(132)
    plt.scatter(df['vol_pct'], df['vix'], alpha=0.5, color='teal')
    plt.title('Volatility vs VIX')
    plt.xlabel('Alpha Volatility (%)')
    plt.ylabel('Alpha VIX (%)')
    plt.grid(True, alpha=0.3)
    
    # 相関係数と傾き（線形回帰）を表示
    if 'vol_pct' in df.columns and 'vix' in df.columns:
        vol_vix_corr = df['vol_pct'].corr(df['vix'])
        x = df['vol_pct'].dropna().values
        y = df['vix'].dropna().values
        if len(x) > 1 and len(y) > 1:
            try:
                from sklearn.linear_model import LinearRegression
                x_reshaped = x.reshape(-1, 1)
                model = LinearRegression().fit(x_reshaped, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                plt.plot(x, intercept + slope * x, 'r--', alpha=0.7)
                plt.text(0.05, 0.95, f'Corr: {vol_vix_corr:.4f}\nSlope: {slope:.4f}', 
                        transform=plt.gca().transAxes, fontsize=9, 
                        verticalalignment='top')
            except:
                plt.text(0.05, 0.95, f'Corr: {vol_vix_corr:.4f}', 
                        transform=plt.gca().transAxes, fontsize=9, 
                        verticalalignment='top')
    
    plt.subplot(133)
    plt.scatter(df['atr_pct'], df['vol_pct'], alpha=0.5, color='darkviolet')
    plt.title('ATR vs Volatility')
    plt.xlabel('Alpha ATR (%)')
    plt.ylabel('Alpha Volatility (%)')
    plt.grid(True, alpha=0.3)
    
    # 相関係数と傾き（線形回帰）を表示
    if 'atr_pct' in df.columns and 'vol_pct' in df.columns:
        atr_vol_corr = df['atr_pct'].corr(df['vol_pct'])
        x = df['atr_pct'].dropna().values
        y = df['vol_pct'].dropna().values
        if len(x) > 1 and len(y) > 1:
            try:
                from sklearn.linear_model import LinearRegression
                x_reshaped = x.reshape(-1, 1)
                model = LinearRegression().fit(x_reshaped, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                plt.plot(x, intercept + slope * x, 'r--', alpha=0.7)
                plt.text(0.05, 0.95, f'Corr: {atr_vol_corr:.4f}\nSlope: {slope:.4f}', 
                        transform=plt.gca().transAxes, fontsize=9, 
                        verticalalignment='top')
            except:
                plt.text(0.05, 0.95, f'Corr: {atr_vol_corr:.4f}', 
                        transform=plt.gca().transAxes, fontsize=9, 
                        verticalalignment='top')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 