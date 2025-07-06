#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Zero Lag EMA with Market-Adaptive UKF チャート描画** 🎯

実際の相場データを使用してZero Lag EMA with MA-UKFインジケーターを
チャートに描画するテストコード。

特徴:
- 設定ファイルからの実データ読み込み
- Zero Lag EMAとMA-UKFフィルタリングの可視化
- 市場レジーム状態の表示
- トレンド信号とMA-UKF信頼度の表示
- 包括的な統計情報の出力
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# データ取得のための依存関係
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError:
    print("Warning: データローダーモジュールが見つかりません。ダミーデータを使用します。")
    DataLoader = None
    CSVDataSource = None
    DataProcessor = None
    BinanceDataSource = None

# インジケーター
try:
    from indicators.zero_lag_ema_ma_ukf import ZeroLagEMAWithMAUKF
    from indicators.price_source import PriceSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators.zero_lag_ema_ma_ukf import ZeroLagEMAWithMAUKF
    from indicators.price_source import PriceSource


class ZeroLagEMAMAUKFChart:
    """
    Zero Lag EMA with Market-Adaptive UKF チャート描画クラス
    
    - ローソク足と出来高
    - MA-UKFフィルタリング済みHLC3価格
    - Zero Lag EMAと通常のEMA
    - トレンド信号
    - 市場レジーム状態
    - MA-UKF信頼度スコア
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.zero_lag_ema = None
        self.result = None
        self.fig = None
        self.axes = None
    
    def generate_dummy_data(self, n_periods: int = 200) -> pd.DataFrame:
        """
        ダミーの相場データを生成
        
        Args:
            n_periods: 生成する期間数
            
        Returns:
            OHLCV形式のDataFrame
        """
        print(f"ダミーデータを生成中... ({n_periods}期間)")
        
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
        
        # 複雑な価格動作をシミュレート
        base_price = 100.0
        price_changes = []
        
        for i in range(n_periods):
            # トレンド成分
            trend = 0.02 * np.sin(i * 0.01) + 0.01 * np.sin(i * 0.005)
            
            # 周期成分
            cycle = 2.0 * np.sin(i * 0.1) + 1.0 * np.sin(i * 0.05)
            
            # ランダムウォーク成分
            random_walk = np.random.normal(0, 0.5)
            
            # ボラティリティクラスタリング
            if i > 0 and abs(price_changes[-1]) > 1.0:
                volatility_multiplier = 1.5
            else:
                volatility_multiplier = 1.0
            
            # レジーム変化（トレンド市場 vs レンジ市場）
            if i % 50 == 0:  # 50期間ごとにレジーム変化
                regime_shock = np.random.choice([-3, 3]) if np.random.random() > 0.7 else 0
            else:
                regime_shock = 0
            
            total_change = trend + cycle + random_walk * volatility_multiplier + regime_shock
            price_changes.append(total_change)
        
        # 価格系列の生成
        prices = [base_price]
        for change in price_changes:
            new_price = max(prices[-1] + change, 1.0)  # 価格が負にならないよう制限
            prices.append(new_price)
        
        prices = np.array(prices[1:])  # 最初の基準価格を除く
        
        # OHLC生成
        high_offset = np.abs(np.random.normal(0, 0.3, n_periods))
        low_offset = np.abs(np.random.normal(0, 0.3, n_periods))
        open_offset = np.random.normal(0, 0.1, n_periods)
        
        ohlc_data = pd.DataFrame({
            'open': prices + open_offset,
            'high': prices + high_offset,
            'low': prices - low_offset,
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n_periods)
        }, index=dates)
        
        # high >= low の制約を満たすよう調整
        ohlc_data['high'] = np.maximum(ohlc_data[['open', 'close']].max(axis=1), ohlc_data['high'])
        ohlc_data['low'] = np.minimum(ohlc_data[['open', 'close']].min(axis=1), ohlc_data['low'])
        
        print(f"ダミーデータ生成完了")
        print(f"期間: {ohlc_data.index.min()} → {ohlc_data.index.max()}")
        print(f"価格範囲: {ohlc_data['close'].min():.2f} - {ohlc_data['close'].max():.2f}")
        
        return ohlc_data
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        if not os.path.exists(config_path):
            print(f"設定ファイルが見つかりません: {config_path}")
            print("ダミーデータを使用します")
            return self.generate_dummy_data()
        
        if DataLoader is None:
            print("データローダーが利用できません。ダミーデータを使用します")
            return self.generate_dummy_data()
        
        try:
            # 設定ファイルの読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # データの準備
            binance_config = config.get('binance_data', {})
            data_dir = binance_config.get('data_dir', 'data/binance')
            binance_data_source = BinanceDataSource(data_dir)
            
            # CSVデータソースはダミーとして渡す
            dummy_csv_source = CSVDataSource("dummy")
            data_loader = DataLoader(
                data_source=dummy_csv_source,
                binance_data_source=binance_data_source
            )
            data_processor = DataProcessor()
            
            # データの読み込みと処理
            print("\nデータを読み込み・処理中...")
            raw_data = data_loader.load_data_from_config(config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(processed_data))
            self.data = processed_data[first_symbol]
            
            print(f"データ読み込み完了: {first_symbol}")
            print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
            print(f"データ数: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            print(f"設定ファイルからのデータ読み込みに失敗: {e}")
            print("ダミーデータを使用します")
            return self.generate_dummy_data()

    def calculate_indicators(self, 
                           ema_period: int = 14,
                           lag_adjustment: float = 1.0,
                           slope_period: int = 1,
                           range_threshold: float = 0.003) -> None:
        """
        Zero Lag EMA with MA-UKFを計算する
        
        Args:
            ema_period: EMA期間
            lag_adjustment: 遅延調整係数
            slope_period: トレンド判定期間
            range_threshold: レンジ判定閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nZero Lag EMA with MA-UKFを計算中...")
        print(f"パラメータ: EMA期間={ema_period}, 遅延調整={lag_adjustment}, "
              f"トレンド期間={slope_period}, レンジ閾値={range_threshold}")
        
        # Zero Lag EMA with MA-UKFの計算
        self.zero_lag_ema = ZeroLagEMAWithMAUKF(
            ema_period=ema_period,
            lag_adjustment=lag_adjustment,
            slope_period=slope_period,
            range_threshold=range_threshold,
            # MA-UKFパラメータ
            ukf_alpha=0.001,
            ukf_beta=2.0,
            ukf_kappa=0.0,
            ukf_base_process_noise=0.001,
            ukf_base_measurement_noise=0.01,
            ukf_volatility_window=10
        )
        
        # インジケーターの計算
        print("計算を実行中...")
        self.result = self.zero_lag_ema.calculate(self.data)
        
        # 計算結果の検証
        valid_zero_lag = self.result.values[~np.isnan(self.result.values)]
        valid_ema = self.result.ema_values[~np.isnan(self.result.ema_values)]
        
        print(f"計算完了!")
        print(f"有効なZero Lag EMA値: {len(valid_zero_lag)}/{len(self.result.values)}")
        print(f"有効な通常EMA値: {len(valid_ema)}/{len(self.result.ema_values)}")
        
        # 基本統計
        if len(valid_zero_lag) > 0:
            print(f"Zero Lag EMA範囲: {valid_zero_lag.min():.2f} - {valid_zero_lag.max():.2f}")
        
        # MA-UKF統計
        if self.result.market_regimes is not None:
            valid_regimes = self.result.market_regimes[~np.isnan(self.result.market_regimes)]
            if len(valid_regimes) > 0:
                print(f"市場レジーム範囲: {valid_regimes.min():.3f} - {valid_regimes.max():.3f}")
        
        if self.result.confidence_scores is not None:
            valid_conf = self.result.confidence_scores[~np.isnan(self.result.confidence_scores)]
            if len(valid_conf) > 0:
                print(f"平均MA-UKF信頼度: {np.mean(valid_conf):.3f}")
            
    def plot(self, 
            title: str = "Zero Lag EMA with Market-Adaptive UKF", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとZero Lag EMA with MA-UKFを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None or self.result is None:
            raise ValueError("データまたはインジケーターが計算されていません。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターデータの取得
        print("チャートデータを準備中...")
        
        # 全データの時系列データフレームを作成
        indicator_df = pd.DataFrame(
            index=self.data.index,
            data={
                'zero_lag_ema': self.result.values,
                'regular_ema': self.result.ema_values,
                'filtered_hlc3': self.result.filtered_source,
                'raw_hlc3': self.result.raw_source,
                'trend_signals': self.result.trend_signals,
                'market_regimes': self.result.market_regimes if self.result.market_regimes is not None else np.nan,
                'confidence_scores': self.result.confidence_scores if self.result.confidence_scores is not None else np.nan
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(indicator_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド方向に基づく色分け
        df['zero_lag_up'] = np.where(df['trend_signals'] == 1, df['zero_lag_ema'], np.nan)
        df['zero_lag_down'] = np.where(df['trend_signals'] == -1, df['zero_lag_ema'], np.nan)
        df['zero_lag_range'] = np.where(df['trend_signals'] == 0, df['zero_lag_ema'], np.nan)
        
        # 市場レジームに基づく背景色（使用しない場合はコメントアウト）
        df['regime_trend'] = np.where(df['market_regimes'] > 0.5, df['market_regimes'], np.nan)
        df['regime_range'] = np.where(np.abs(df['market_regimes']) < 0.3, df['market_regimes'], np.nan)
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 1. MA-UKFフィルタリング済みHLC3
        main_plots.append(mpf.make_addplot(df['filtered_hlc3'], color='cyan', width=1, 
                                         alpha=0.7, label='Filtered HLC3'))
        
        # 2. 通常のEMA
        main_plots.append(mpf.make_addplot(df['regular_ema'], color='blue', width=1.5, 
                                         alpha=0.8, label='Regular EMA'))
        
        # 3. Zero Lag EMA（トレンド方向別）
        main_plots.append(mpf.make_addplot(df['zero_lag_up'], color='green', width=2.5, 
                                         label='Zero Lag EMA (Up)'))
        main_plots.append(mpf.make_addplot(df['zero_lag_down'], color='red', width=2.5, 
                                         label='Zero Lag EMA (Down)'))
        main_plots.append(mpf.make_addplot(df['zero_lag_range'], color='gray', width=2.5, 
                                         label='Zero Lag EMA (Range)'))
        
        # 4. トレンド信号パネル
        trend_panel = 1 if show_volume else 1
        trend_plot = mpf.make_addplot(df['trend_signals'], panel=trend_panel, color='orange', 
                                    width=1.5, ylabel='Trend Signals', secondary_y=False, 
                                    label='Trend', type='line')
        
        # 5. 市場レジームパネル（MA-UKF固有）
        regime_panel = 2 if show_volume else 2
        regime_plot = mpf.make_addplot(df['market_regimes'], panel=regime_panel, color='purple', 
                                     width=1.5, ylabel='Market Regimes', secondary_y=False, 
                                     label='Regimes', type='line')
        
        # 6. MA-UKF信頼度パネル
        conf_panel = 3 if show_volume else 3
        conf_plot = mpf.make_addplot(df['confidence_scores'], panel=conf_panel, color='darkgreen', 
                                   width=1.2, ylabel='MA-UKF Confidence', secondary_y=False, 
                                   label='Confidence', type='line')
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        # パネル比率の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1)  # メイン:出来高:トレンド:レジーム:信頼度
            # 出来高を表示する場合はパネル番号を調整
            trend_plot = mpf.make_addplot(df['trend_signals'], panel=2, color='orange', 
                                        width=1.5, ylabel='Trend Signals', secondary_y=False, 
                                        label='Trend', type='line')
            regime_plot = mpf.make_addplot(df['market_regimes'], panel=3, color='purple', 
                                         width=1.5, ylabel='Market Regimes', secondary_y=False, 
                                         label='Regimes', type='line')
            conf_plot = mpf.make_addplot(df['confidence_scores'], panel=4, color='darkgreen', 
                                       width=1.2, ylabel='MA-UKF Confidence', secondary_y=False, 
                                       label='Confidence', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1, 1, 1)  # メイン:トレンド:レジーム:信頼度
        
        # すべてのプロットを結合
        all_plots = main_plots + [trend_plot, regime_plot, conf_plot]
        kwargs['addplot'] = all_plots
        
        # データの検証
        if df['zero_lag_ema'].isna().all():
            print("⚠️ 警告: Zero Lag EMA値がすべてNaNです。パラメータまたはデータを確認してください。")
            return
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Filtered HLC3', 'Regular EMA', 'Zero Lag EMA (Up)', 
                       'Zero Lag EMA (Down)', 'Zero Lag EMA (Range)'], 
                      loc='upper left', fontsize=9)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # トレンド信号パネル
        trend_ax = axes[1 + panel_offset]
        trend_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        trend_ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        trend_ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        trend_ax.set_ylim(-1.5, 1.5)
        
        # 市場レジームパネル
        regime_ax = axes[2 + panel_offset]
        regime_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        regime_ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Trend Threshold')
        regime_ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
        regime_ax.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Range Threshold')
        regime_ax.axhline(y=-0.3, color='gray', linestyle=':', alpha=0.5)
        regime_ax.set_ylim(-1.2, 1.2)
        
        # 信頼度パネル
        conf_ax = axes[3 + panel_offset]
        conf_ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High Confidence')
        conf_ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
        conf_ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Low Confidence')
        conf_ax.set_ylim(0, 1)
        
        # 統計情報の表示
        self.print_statistics(df)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def print_statistics(self, df: pd.DataFrame) -> None:
        """統計情報を表示"""
        print(f"\n{'='*60}")
        print(f"📊 Zero Lag EMA with MA-UKF 統計情報")
        print(f"{'='*60}")
        
        # 基本データ統計
        total_points = len(df)
        valid_zero_lag = df['zero_lag_ema'].dropna()
        valid_ema = df['regular_ema'].dropna()
        
        print(f"🔸 データ統計:")
        print(f"  総データ点数: {total_points}")
        print(f"  有効Zero Lag EMA: {len(valid_zero_lag)} ({len(valid_zero_lag)/total_points*100:.1f}%)")
        print(f"  有効Regular EMA: {len(valid_ema)} ({len(valid_ema)/total_points*100:.1f}%)")
        
        # 価格統計
        if len(valid_zero_lag) > 0:
            print(f"\n🔸 Zero Lag EMA統計:")
            print(f"  範囲: {valid_zero_lag.min():.2f} - {valid_zero_lag.max():.2f}")
            print(f"  平均: {valid_zero_lag.mean():.2f}")
            print(f"  標準偏差: {valid_zero_lag.std():.2f}")
        
        # トレンド統計
        trend_counts = df['trend_signals'].value_counts()
        print(f"\n🔸 トレンド分析:")
        for trend_val in [1, -1, 0]:
            if trend_val in trend_counts:
                count = trend_counts[trend_val]
                percentage = count / total_points * 100
                trend_name = {1: "上昇", -1: "下降", 0: "レンジ"}[trend_val]
                print(f"  {trend_name}トレンド: {count} ({percentage:.1f}%)")
        
        # MA-UKF統計
        if 'market_regimes' in df.columns and not df['market_regimes'].isna().all():
            valid_regimes = df['market_regimes'].dropna()
            if len(valid_regimes) > 0:
                print(f"\n🔸 市場レジーム分析:")
                print(f"  平均レジーム値: {valid_regimes.mean():.3f}")
                print(f"  レジーム範囲: {valid_regimes.min():.3f} - {valid_regimes.max():.3f}")
                
                trend_market = (valid_regimes > 0.5).sum()
                range_market = (np.abs(valid_regimes) < 0.3).sum()
                print(f"  トレンド市場: {trend_market} ({trend_market/len(valid_regimes)*100:.1f}%)")
                print(f"  レンジ市場: {range_market} ({range_market/len(valid_regimes)*100:.1f}%)")
        
        if 'confidence_scores' in df.columns and not df['confidence_scores'].isna().all():
            valid_conf = df['confidence_scores'].dropna()
            if len(valid_conf) > 0:
                print(f"\n🔸 MA-UKF信頼度:")
                print(f"  平均信頼度: {valid_conf.mean():.3f}")
                print(f"  信頼度範囲: {valid_conf.min():.3f} - {valid_conf.max():.3f}")
                
                high_conf = (valid_conf > 0.8).sum()
                low_conf = (valid_conf < 0.4).sum()
                print(f"  高信頼度(>0.8): {high_conf} ({high_conf/len(valid_conf)*100:.1f}%)")
                print(f"  低信頼度(<0.4): {low_conf} ({low_conf/len(valid_conf)*100:.1f}%)")
        
        # フィルタリング効果
        if 'filtered_hlc3' in df.columns and 'raw_hlc3' in df.columns:
            valid_filtered = df['filtered_hlc3'].dropna()
            valid_raw = df['raw_hlc3'].dropna()
            
            if len(valid_filtered) > 1 and len(valid_raw) > 1:
                min_len = min(len(valid_filtered), len(valid_raw))
                filtered_vol = np.std(np.diff(valid_filtered.iloc[:min_len]))
                raw_vol = np.std(np.diff(valid_raw.iloc[:min_len]))
                
                if raw_vol > 0:
                    noise_reduction = (1.0 - filtered_vol / raw_vol) * 100
                    print(f"\n🔸 フィルタリング効果:")
                    print(f"  ノイズ除去率: {noise_reduction:.1f}%")
                    print(f"  元の変動性: {raw_vol:.4f}")
                    print(f"  フィルター後: {filtered_vol:.4f}")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Zero Lag EMA with MA-UKFチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', 
                       help='設定ファイルのパス（存在しない場合はダミーデータを使用）')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--ema-period', type=int, default=14, help='EMA期間 (デフォルト: 14)')
    parser.add_argument('--lag-adjustment', type=float, default=2.0, help='遅延調整係数 (デフォルト: 2.0)')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示にする')
    args = parser.parse_args()
    
    print("🎯 Zero Lag EMA with Market-Adaptive UKF チャートテスト")
    print("=" * 60)
    
    # チャートを作成
    chart = ZeroLagEMAMAUKFChart()
    
    try:
        # データ読み込み
        chart.load_data_from_config(args.config)
        
        # インジケーター計算
        chart.calculate_indicators(
            ema_period=args.ema_period,
            lag_adjustment=args.lag_adjustment
        )
        
        # チャート描画
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            show_volume=not args.no_volume,
            savefig=args.output
        )
        
        print(f"\n✅ チャート描画完了!")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 