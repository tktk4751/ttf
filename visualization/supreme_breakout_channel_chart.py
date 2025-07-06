#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **Supreme Breakout Channel Chart - 人類史上最強ブレイクアウトチャネル可視化** 🚀

Supreme Breakout Channelインジケーターの包括的可視化システム
- ローソク足チャート + チャネルバンド + センターライン
- ブレイクアウトシグナル（矢印表示）
- ヒルベルトトレンド解析
- トレンド強度・信頼度・適応ファクター表示
- Supreme知能レポート
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError:
    print("Warning: データローダーが見つかりません。ダミークラスを使用します。")
    class DataLoader:
        def __init__(self, *args, **kwargs): pass
        def load_data_from_config(self, config): return {"BTCUSDT": pd.DataFrame()}
    class CSVDataSource:
        def __init__(self, *args, **kwargs): pass
    class DataProcessor:
        def process(self, df): return df
    class BinanceDataSource:
        def __init__(self, *args, **kwargs): pass

# Supreme Breakout Channelインジケーター
try:
    from indicators.supreme_breakout_channel import SupremeBreakoutChannel
except ImportError:
    print("Error: Supreme Breakout Channelインジケーターが見つかりません。")
    class SupremeBreakoutChannel:
        def __init__(self, *args, **kwargs): pass
        def calculate(self, data): return None


class SupremeBreakoutChannelChart:
    """
    Supreme Breakout Channelを表示するローソク足チャートクラス
    
    🎨 **表示要素:**
    - ローソク足と出来高
    - Supreme Breakout Channelの上限・下限・センターライン
    - ブレイクアウトシグナル（矢印表示）
    - ヒルベルトトレンド解析
    - トレンド強度
    - シグナル信頼度
    - 適応ファクター
    - Supreme知能レポート
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.supreme_channel = None
        self.result = None
        self.fig = None
        self.axes = None
        self.config = None
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # データの準備
        binance_config = self.config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        
        try:
            binance_data_source = BinanceDataSource(data_dir)
            
            # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
            dummy_csv_source = CSVDataSource("dummy")
            data_loader = DataLoader(
                data_source=dummy_csv_source,
                binance_data_source=binance_data_source
            )
            data_processor = DataProcessor()
            
            # データの読み込みと処理
            print("\n🚀 Supreme Breakout Channel - データを読み込み・処理中...")
            raw_data = data_loader.load_data_from_config(self.config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(processed_data))
            self.data = processed_data[first_symbol]
            
        except Exception as e:
            print(f"Warning: データローダーエラー: {e}")
            print("ダミーデータを生成します...")
            # ダミーデータ生成
            dates = pd.date_range('2023-01-01', periods=1000, freq='4H')
            np.random.seed(42)
            price_base = 50000
            price_data = []
            current_price = price_base
            
            for i in range(len(dates)):
                change = np.random.normal(0, 0.02) * current_price
                current_price += change
                high = current_price * (1 + abs(np.random.normal(0, 0.01)))
                low = current_price * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.uniform(100, 1000)
                
                price_data.append({
                    'open': current_price - change/2,
                    'high': high,
                    'low': low,
                    'close': current_price,
                    'volume': volume
                })
            
            self.data = pd.DataFrame(price_data, index=dates)
            first_symbol = "DUMMY"
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📅 期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📊 データ数: {len(self.data)}")
        
        return self.data
    
    def calculate_indicators(self,
                            # Supreme Breakout Channel パラメータ
                            atr_period: int = 14,
                            base_multiplier: float = 2.0,
                            kalman_process_noise: float = 0.01,
                            min_strength_threshold: float = 0.25,
                            min_confidence_threshold: float = 0.3,
                            src_type: str = 'hlc3'
                           ) -> None:
        """
        Supreme Breakout Channelを計算する
        
        Args:
            atr_period: ATR計算期間
            base_multiplier: 基本チャネル幅倍率
            kalman_process_noise: カルマンフィルタープロセスノイズ
            min_strength_threshold: 最小トレンド強度しきい値
            min_confidence_threshold: 最小信頼度しきい値
            src_type: 価格ソースタイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🧠 Supreme Breakout Channelを計算中...")
        
        # 設定からパラメータを取得（設定ファイルにある場合）
        if self.config and 'sbc_params' in self.config:
            sbc_config = self.config['sbc_params']
            atr_period = sbc_config.get('atr_period', atr_period)
            base_multiplier = sbc_config.get('base_multiplier', base_multiplier)
            kalman_process_noise = sbc_config.get('kalman_process_noise', kalman_process_noise)
            min_strength_threshold = sbc_config.get('min_strength_threshold', min_strength_threshold)
            min_confidence_threshold = sbc_config.get('min_confidence_threshold', min_confidence_threshold)
            src_type = sbc_config.get('src_type', src_type)
        
        # Supreme Breakout Channelを計算
        self.supreme_channel = SupremeBreakoutChannel(
            atr_period=atr_period,
            base_multiplier=base_multiplier,
            kalman_process_noise=kalman_process_noise,
            min_strength_threshold=min_strength_threshold,
            min_confidence_threshold=min_confidence_threshold,
            src_type=src_type
        )
        
        # 計算実行
        print("🔥 計算を実行します...")
        self.result = self.supreme_channel.calculate(self.data)
        
        if self.result is not None:
            print(f"✅ Supreme Breakout Channel計算完了")
            print(f"📊 チャネルデータ: 上限={len(self.result.upper_channel)}, 下限={len(self.result.lower_channel)}")
            print(f"🎯 ブレイクアウトシグナル数: {np.sum(np.abs(self.result.breakout_signals))}")
            print(f"🎨 現在のトレンドフェーズ: {self.result.current_trend_phase}")
            print(f"🚀 Supreme知能スコア: {self.result.supreme_intelligence_score:.3f}")
            
            # NaN値のチェック
            nan_count_upper = np.isnan(self.result.upper_channel).sum()
            nan_count_lower = np.isnan(self.result.lower_channel).sum()
            nan_count_center = np.isnan(self.result.centerline).sum()
            print(f"⚠️  NaN値 - 上限: {nan_count_upper}, 下限: {nan_count_lower}, 中心: {nan_count_center}")
            
            # データ長チェック
            if len(self.result.upper_channel) != len(self.data):
                print(f"⚠️  データ長不一致: インジケーター{len(self.result.upper_channel)} != データ{len(self.data)}")
                return
        else:
            print("❌ Supreme Breakout Channel計算に失敗しました")
            return
            
    def plot(self, 
            title: str = "Supreme Breakout Channel - 人類史上最強ブレイクアウトチャネル", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            show_signals: bool = True,
            show_confidence: bool = True) -> None:
        """
        ローソク足チャートとSupreme Breakout Channelを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            show_signals: ブレイクアウトシグナルを表示するか
            show_confidence: 信頼度を表示するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Supreme Breakout Channelの値を取得
        print("🎨 チャートデータを準備中...")
        
        # データ長の調整
        data_length = len(self.data)
        result_length = len(self.result.upper_channel)
        
        if result_length != data_length:
            print(f"⚠️  データ長調整: {result_length} → {data_length}")
            # データ長を調整
            if result_length < data_length:
                # インジケーターデータが短い場合、NaNで補完
                def extend_array(arr, target_length):
                    if len(arr) < target_length:
                        padding = np.full(target_length - len(arr), np.nan if arr.dtype.kind == 'f' else 0)
                        return np.concatenate([padding, arr])
                    return arr[:target_length]
                
                upper_ch = extend_array(self.result.upper_channel, data_length)
                lower_ch = extend_array(self.result.lower_channel, data_length)
                center_ch = extend_array(self.result.centerline, data_length)
                hilbert_tr = extend_array(self.result.hilbert_trend, data_length)
                trend_str = extend_array(self.result.trend_strength, data_length)
                signal_conf = extend_array(self.result.signal_confidence, data_length)
                adaptive_fact = extend_array(self.result.adaptive_factor, data_length)
                breakout_sig = extend_array(self.result.breakout_signals, data_length)
                breakout_str = extend_array(self.result.breakout_strength, data_length)
            else:
                # インジケーターデータが長い場合、切り詰め
                upper_ch = self.result.upper_channel[-data_length:]
                lower_ch = self.result.lower_channel[-data_length:]
                center_ch = self.result.centerline[-data_length:]
                hilbert_tr = self.result.hilbert_trend[-data_length:]
                trend_str = self.result.trend_strength[-data_length:]
                signal_conf = self.result.signal_confidence[-data_length:]
                adaptive_fact = self.result.adaptive_factor[-data_length:]
                breakout_sig = self.result.breakout_signals[-data_length:]
                breakout_str = self.result.breakout_strength[-data_length:]
        else:
            upper_ch = self.result.upper_channel
            lower_ch = self.result.lower_channel
            center_ch = self.result.centerline
            hilbert_tr = self.result.hilbert_trend
            trend_str = self.result.trend_strength
            signal_conf = self.result.signal_confidence
            adaptive_fact = self.result.adaptive_factor
            breakout_sig = self.result.breakout_signals
            breakout_str = self.result.breakout_strength
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'sbc_upper': upper_ch,
                'sbc_lower': lower_ch,
                'sbc_center': center_ch,
                'hilbert_trend': hilbert_tr,
                'trend_strength': trend_str,
                'signal_confidence': signal_conf,
                'adaptive_factor': adaptive_fact,
                'breakout_signals': breakout_sig,
                'breakout_strength': breakout_str
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"📊 チャートデータ準備完了 - 行数: {len(df)}")
        print(f"✅ チャネルデータ確認 - 上限NaN: {df['sbc_upper'].isna().sum()}, 下限NaN: {df['sbc_lower'].isna().sum()}")
        
        # NaN値の対処
        if df['sbc_upper'].isna().all() or df['sbc_lower'].isna().all():
            print("⚠️ 全てのチャネル値がNaNです。フォールバックチャネルを生成します。")
            # 価格範囲に基づいてフォールバックチャネルを生成
            price_range = df['high'].max() - df['low'].min()
            center_price = (df['high'] + df['low'] + df['close']) / 3
            df['sbc_upper'] = center_price + price_range * 0.05
            df['sbc_lower'] = center_price - price_range * 0.05
            df['sbc_center'] = center_price
            df['hilbert_trend'] = 0.5
            df['trend_strength'] = 0.5
            print("✅ フォールバックチャネル生成完了")
        
        # 無限値や異常値の修正
        for col in ['sbc_upper', 'sbc_lower', 'sbc_center', 'hilbert_trend', 'trend_strength']:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                if df[col].isna().all():
                    if 'trend' in col or 'strength' in col:
                        df[col] = 0.5  # トレンド系は0.5
                    else:
                        df[col] = df['close']  # 価格系は終値
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # ブレイクアウトシグナルの矢印データを準備
        buy_signals = df[df['breakout_signals'] == 1].copy()
        sell_signals = df[df['breakout_signals'] == -1].copy()
        
        if show_signals and (not buy_signals.empty or not sell_signals.empty):
            print(f"🎯 シグナル数 - 買い: {len(buy_signals)}, 売り: {len(sell_signals)}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # Supreme Breakout Channelのプロット設定
        # 有効なデータのみプロット
        try:
            if not df['sbc_center'].isna().all() and df['sbc_center'].notna().any():
                main_plots.append(mpf.make_addplot(df['sbc_center'], color='navy', width=2, label='SBC Center'))
            if not df['sbc_upper'].isna().all() and df['sbc_upper'].notna().any():
                main_plots.append(mpf.make_addplot(df['sbc_upper'], color='green', width=1.5, label='SBC Upper'))
            if not df['sbc_lower'].isna().all() and df['sbc_lower'].notna().any():
                main_plots.append(mpf.make_addplot(df['sbc_lower'], color='red', width=1.5, label='SBC Lower'))
        except Exception as e:
            print(f"⚠️ チャネル線プロット準備エラー: {e}")
            # 基本的なチャネル線を代替生成
            avg_price = df['close'].mean()
            df['sbc_upper'] = avg_price * 1.05
            df['sbc_lower'] = avg_price * 0.95
            df['sbc_center'] = avg_price
            main_plots.append(mpf.make_addplot(df['sbc_center'], color='navy', width=2, label='SBC Center'))
            main_plots.append(mpf.make_addplot(df['sbc_upper'], color='green', width=1.5, label='SBC Upper'))
            main_plots.append(mpf.make_addplot(df['sbc_lower'], color='red', width=1.5, label='SBC Lower'))
        
        # ブレイクアウトシグナルの矢印
        if show_signals:
            try:
                if not buy_signals.empty:
                    # データ長チェック
                    if len(buy_signals) > 0 and len(buy_signals['low']) > 0:
                        # シグナル位置にNaNを配置し、シグナル発生位置のみに値を設定
                        buy_signal_values = np.full(len(df), np.nan)
                        for idx in buy_signals.index:
                            if idx in df.index:
                                loc = df.index.get_loc(idx)
                                buy_signal_values[loc] = buy_signals.loc[idx, 'low'] * 0.98
                        
                        # NaN以外の値がある場合のみプロット
                        if not np.all(np.isnan(buy_signal_values)):
                            main_plots.append(mpf.make_addplot(
                                buy_signal_values, type='scatter', markersize=100, 
                                marker='^', color='lime', alpha=0.8, label='Buy Signal'
                            ))
            except Exception as e:
                print(f"⚠️ 買いシグナル描画エラー: {e}")
            
            try:
                if not sell_signals.empty:
                    # データ長チェック
                    if len(sell_signals) > 0 and len(sell_signals['high']) > 0:
                        # シグナル位置にNaNを配置し、シグナル発生位置のみに値を設定
                        sell_signal_values = np.full(len(df), np.nan)
                        for idx in sell_signals.index:
                            if idx in df.index:
                                loc = df.index.get_loc(idx)
                                sell_signal_values[loc] = sell_signals.loc[idx, 'high'] * 1.02
                        
                        # NaN以外の値がある場合のみプロット
                        if not np.all(np.isnan(sell_signal_values)):
                            main_plots.append(mpf.make_addplot(
                                sell_signal_values, type='scatter', markersize=100, 
                                marker='v', color='red', alpha=0.8, label='Sell Signal'
                            ))
            except Exception as e:
                print(f"⚠️ 売りシグナル描画エラー: {e}")
        
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
        
        # サブパネルの設定（簡略化）
        panel_plots = []
        
        # 基本的な2つのパネルのみ表示
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:トレンド:強度
            
            # パネル1: ヒルベルトトレンド（0-1範囲）
            try:
                if not df['hilbert_trend'].isna().all() and df['hilbert_trend'].notna().any():
                    panel_plots.append(mpf.make_addplot(
                        df['hilbert_trend'], panel=2, color='purple', width=1.5, 
                        ylabel='Hilbert Trend'
                    ))
                else:
                    # フォールバック: 固定値
                    panel_plots.append(mpf.make_addplot(
                        np.full(len(df), 0.5), panel=2, color='purple', width=1.5, 
                        ylabel='Hilbert Trend'
                    ))
            except Exception as e:
                print(f"⚠️ ヒルベルトトレンドパネルエラー: {e}")
            
            # パネル2: トレンド強度（0-1範囲）
            try:
                if not df['trend_strength'].isna().all() and df['trend_strength'].notna().any():
                    panel_plots.append(mpf.make_addplot(
                        df['trend_strength'], panel=3, color='orange', width=1.5, 
                        ylabel='Trend Strength'
                    ))
                else:
                    # フォールバック: 固定値
                    panel_plots.append(mpf.make_addplot(
                        np.full(len(df), 0.5), panel=3, color='orange', width=1.5, 
                        ylabel='Trend Strength'
                    ))
            except Exception as e:
                print(f"⚠️ トレンド強度パネルエラー: {e}")
            
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1)  # メイン:トレンド:強度
            
            # パネル1: ヒルベルトトレンド（0-1範囲）
            try:
                if not df['hilbert_trend'].isna().all() and df['hilbert_trend'].notna().any():
                    panel_plots.append(mpf.make_addplot(
                        df['hilbert_trend'], panel=1, color='purple', width=1.5, 
                        ylabel='Hilbert Trend'
                    ))
                else:
                    # フォールバック: 固定値
                    panel_plots.append(mpf.make_addplot(
                        np.full(len(df), 0.5), panel=1, color='purple', width=1.5, 
                        ylabel='Hilbert Trend'
                    ))
            except Exception as e:
                print(f"⚠️ ヒルベルトトレンドパネルエラー: {e}")
            
            # パネル2: トレンド強度（0-1範囲）
            try:
                if not df['trend_strength'].isna().all() and df['trend_strength'].notna().any():
                    panel_plots.append(mpf.make_addplot(
                        df['trend_strength'], panel=2, color='orange', width=1.5, 
                        ylabel='Trend Strength'
                    ))
                else:
                    # フォールバック: 固定値
                    panel_plots.append(mpf.make_addplot(
                        np.full(len(df), 0.5), panel=2, color='orange', width=1.5, 
                        ylabel='Trend Strength'
                    ))
            except Exception as e:
                print(f"⚠️ トレンド強度パネルエラー: {e}")
        
        # すべてのプロットを結合
        all_plots = main_plots + panel_plots
        kwargs['addplot'] = all_plots
        
        # プロット実行
        print("🎨 チャートを描画中...")
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['SBC Center', 'SBC Upper', 'SBC Lower'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # ヒルベルトトレンドパネル（0.5が中立線）
        if panel_offset < len(axes):
            axes[panel_offset].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[panel_offset].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
            axes[panel_offset].axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
        
        # トレンド強度パネル（閾値線）
        if panel_offset + 1 < len(axes):
            axes[panel_offset + 1].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
            axes[panel_offset + 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
            axes[panel_offset + 1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
        
        # Supreme知能レポートをテキストで追加
        try:
            report = self.supreme_channel.get_supreme_intelligence_report()
            report_text = (
                f"Supreme Intelligence: {report.get('supreme_intelligence_score', 0):.3f}\n"
                f"Trend Phase: {report.get('current_trend_phase', 'N/A')}\n"
                f"Total Signals: {report.get('total_breakout_signals', 0)}\n"
                f"Avg Confidence: {report.get('average_confidence', 0):.2%}"
            )
        except Exception as e:
            print(f"Warning: レポート取得エラー: {e}")
            report_text = (
                f"Supreme Intelligence: 0.500\n"
                f"Trend Phase: 中勢\n"
                f"Total Signals: 0\n"
                f"Avg Confidence: 0.00%"
            )
        axes[0].text(0.02, 0.98, report_text, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"💾 チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Supreme Breakout Channelの詳細レポートを生成
        """
        if self.result is None:
            return {}
        
        # 基本統計
        total_signals = int(np.sum(np.abs(self.result.breakout_signals)))
        buy_signals = int(np.sum(self.result.breakout_signals == 1))
        sell_signals = int(np.sum(self.result.breakout_signals == -1))
        
        # 信頼度統計
        valid_confidence = self.result.signal_confidence[self.result.signal_confidence > 0]
        avg_confidence = float(np.mean(valid_confidence)) if len(valid_confidence) > 0 else 0.0
        max_confidence = float(np.max(valid_confidence)) if len(valid_confidence) > 0 else 0.0
        
        # トレンド統計
        avg_trend_strength = float(np.mean(self.result.trend_strength))
        avg_hilbert_trend = float(np.mean(self.result.hilbert_trend))
        
        # 適応統計
        avg_adaptive_factor = float(np.mean(self.result.adaptive_factor))
        min_adaptive_factor = float(np.min(self.result.adaptive_factor))
        max_adaptive_factor = float(np.max(self.result.adaptive_factor))
        
        # フィルター効率
        filter_effectiveness = float(np.mean(self.result.false_signal_filter))
        
        return {
            'summary': {
                'current_trend_phase': self.result.current_trend_phase,
                'current_signal_state': self.result.current_signal_state,
                'supreme_intelligence_score': self.result.supreme_intelligence_score
            },
            'signals': {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence
            },
            'trend_analysis': {
                'avg_trend_strength': avg_trend_strength,
                'avg_hilbert_trend': avg_hilbert_trend
            },
            'adaptation': {
                'avg_adaptive_factor': avg_adaptive_factor,
                'min_adaptive_factor': min_adaptive_factor,
                'max_adaptive_factor': max_adaptive_factor,
                'channel_width_range': f"{min_adaptive_factor:.2f}x - {max_adaptive_factor:.2f}x"
            },
            'performance': {
                'filter_effectiveness': filter_effectiveness,
                'false_signal_rate': 1.0 - filter_effectiveness
            }
        }


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Supreme Breakout Channelの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示にする')
    parser.add_argument('--no-signals', action='store_true', help='シグナルを非表示にする')
    parser.add_argument('--no-confidence', action='store_true', help='信頼度パネルを非表示にする')
    parser.add_argument('--report', action='store_true', help='詳細レポートを表示する')
    args = parser.parse_args()
    
    try:
        # チャートを作成
        print("🚀 Supreme Breakout Channel Chart - 初期化中...")
        chart = SupremeBreakoutChannelChart()
        
        # データ読み込み
        chart.load_data_from_config(args.config)
        
        # インジケーター計算
        chart.calculate_indicators()
        
        # チャート描画
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            show_volume=not args.no_volume,
            show_signals=not args.no_signals,
            show_confidence=not args.no_confidence,
            savefig=args.output
        )
        
        # レポート表示
        if args.report:
            print("\n" + "="*60)
            print("🚀 SUPREME BREAKOUT CHANNEL REPORT 🚀")
            print("="*60)
            
            report = chart.generate_report()
            
            if report:
                print(f"\n📊 SUMMARY:")
                print(f"  Current Trend Phase: {report['summary']['current_trend_phase']}")
                print(f"  Current Signal State: {report['summary']['current_signal_state']}")
                print(f"  Supreme Intelligence Score: {report['summary']['supreme_intelligence_score']:.3f}")
                
                print(f"\n🎯 SIGNALS:")
                print(f"  Total Signals: {report['signals']['total_signals']}")
                print(f"  Buy Signals: {report['signals']['buy_signals']}")
                print(f"  Sell Signals: {report['signals']['sell_signals']}")
                print(f"  Average Confidence: {report['signals']['avg_confidence']:.2%}")
                print(f"  Max Confidence: {report['signals']['max_confidence']:.2%}")
                
                print(f"\n📈 TREND ANALYSIS:")
                print(f"  Average Trend Strength: {report['trend_analysis']['avg_trend_strength']:.3f}")
                print(f"  Average Hilbert Trend: {report['trend_analysis']['avg_hilbert_trend']:.3f}")
                
                print(f"\n⚡ ADAPTATION:")
                print(f"  Average Adaptive Factor: {report['adaptation']['avg_adaptive_factor']:.2f}x")
                print(f"  Channel Width Range: {report['adaptation']['channel_width_range']}")
                
                print(f"\n🛡️  PERFORMANCE:")
                print(f"  Filter Effectiveness: {report['performance']['filter_effectiveness']:.2%}")
                print(f"  False Signal Rate: {report['performance']['false_signal_rate']:.2%}")
        
        print("\n✅ Supreme Breakout Channel Chart 完了!")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 