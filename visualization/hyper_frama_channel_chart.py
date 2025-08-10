#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.hyper_frama_channel import HyperFRAMAChannel


class HyperFRAMAChannelChart:
    """
    HyperFRAMAChannelを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - HyperFRAMAミッドライン
    - 上限・下限チャネルバンド
    - ATR値とボラティリティレジーム
    - 動的乗数とHyperER値
    - チャネル幅パーセンタイル
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_frama_channel = None
        self.channel_result = None
        self.fig = None
        self.axes = None
    
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

    def calculate_indicators(self,
                            period: int = 14,
                            multiplier_mode: str = "dynamic",
                            fixed_multiplier: float = 2.5,
                            src_type: str = "hlc3",
                            # HyperFRAMAパラメータ
                            hyper_frama_period: int = 16,
                            hyper_frama_src_type: str = 'hl2',
                            hyper_frama_fc: int = 1,
                            hyper_frama_sc: int = 198,
                            hyper_frama_alpha_multiplier: float = 0.5,
                            # X_ATRパラメータ
                            x_atr_period: float = 12.0,
                            x_atr_tr_method: str = 'str',
                            x_atr_smoother_type: str = 'frama',
                            # HyperERパラメータ
                            hyper_er_period: int = 8,
                            hyper_er_midline_period: int = 100,
                            # チャネル独自パラメータ
                            enable_signals: bool = True,
                            enable_percentile: bool = True,
                            percentile_period: int = 100
                           ) -> None:
        """
        HyperFRAMAChannelを計算する
        
        Args:
            period: 基本期間
            multiplier_mode: 乗数モード ("fixed" or "dynamic")
            fixed_multiplier: 固定乗数値
            src_type: 価格ソースタイプ
            hyper_frama_period: HyperFRAMA期間
            hyper_frama_src_type: HyperFRAMAソースタイプ
            hyper_frama_fc: Fast Constant
            hyper_frama_sc: Slow Constant
            hyper_frama_alpha_multiplier: アルファ調整係数
            x_atr_period: X_ATR期間
            x_atr_tr_method: TRメソッド
            x_atr_smoother_type: スムーザータイプ
            hyper_er_period: HyperER期間
            hyper_er_midline_period: HyperERミッドライン期間
            enable_signals: シグナル有効化
            enable_percentile: パーセンタイル分析有効化
            percentile_period: パーセンタイル期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nHyperFRAMAChannelを計算中...")
        
        # HyperFRAMAChannelを計算
        self.hyper_frama_channel = HyperFRAMAChannel(
            period=period,
            multiplier_mode=multiplier_mode,
            fixed_multiplier=fixed_multiplier,
            src_type=src_type,
            # HyperFRAMAパラメータ
            hyper_frama_period=hyper_frama_period,
            hyper_frama_src_type=hyper_frama_src_type,
            hyper_frama_fc=hyper_frama_fc,
            hyper_frama_sc=hyper_frama_sc,
            hyper_frama_alpha_multiplier=hyper_frama_alpha_multiplier,
            # X_ATRパラメータ
            x_atr_period=x_atr_period,
            x_atr_tr_method=x_atr_tr_method,
            x_atr_smoother_type=x_atr_smoother_type,
            # HyperERパラメータ
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            # チャネル独自パラメータ
            enable_signals=enable_signals,
            enable_percentile=enable_percentile,
            percentile_period=percentile_period
        )
        
        # HyperFRAMAChannelの計算
        print("計算を実行します...")
        self.channel_result = self.hyper_frama_channel.calculate(self.data)
        
        # 結果の取得テスト
        midline = self.channel_result.midline
        upper_band = self.channel_result.upper_band
        lower_band = self.channel_result.lower_band
        atr_values = self.channel_result.atr_values
        multiplier_values = self.channel_result.multiplier_values
        
        print(f"計算完了 - ミッドライン: {len(midline)}, 上限バンド: {len(upper_band)}, 下限バンド: {len(lower_band)}")
        print(f"ATR: {len(atr_values)}, 乗数: {len(multiplier_values)}")
        
        # NaN値のチェック
        nan_count_midline = np.isnan(midline).sum()
        nan_count_upper = np.isnan(upper_band).sum()
        nan_count_lower = np.isnan(lower_band).sum()
        nan_count_atr = np.isnan(atr_values).sum()
        print(f"NaN値 - ミッドライン: {nan_count_midline}, 上限: {nan_count_upper}, 下限: {nan_count_lower}, ATR: {nan_count_atr}")
        
        print("HyperFRAMAChannel計算完了")
            
    def plot(self, 
            title: str = "HyperFRAMA Channel", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとHyperFRAMAChannelを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.hyper_frama_channel is None or self.channel_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # HyperFRAMAChannelの値を取得
        print("HyperFRAMAChannelデータを取得中...")
        midline = self.channel_result.midline
        upper_band = self.channel_result.upper_band
        lower_band = self.channel_result.lower_band
        bandwidth = self.channel_result.bandwidth
        atr_values = self.channel_result.atr_values
        multiplier_values = self.channel_result.multiplier_values
        er_values = self.channel_result.er_values
        channel_position = self.channel_result.channel_position
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'midline': midline,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'bandwidth': bandwidth,
                'atr_values': atr_values,
                'multiplier_values': multiplier_values,
                'channel_position': channel_position
            }
        )
        
        # HyperER値がある場合は追加
        if er_values is not None:
            full_df['er_values'] = er_values
        
        # パーセンタイル分析がある場合は追加
        if self.channel_result.channel_width_percentile is not None:
            full_df['channel_width_percentile'] = self.channel_result.channel_width_percentile
        
        # ボラティリティレジームがある場合は追加
        if self.channel_result.volatility_regime is not None:
            full_df['volatility_regime'] = self.channel_result.volatility_regime
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"チャネルデータ確認 - ミッドライン NaN: {df['midline'].isna().sum()}, 上限バンド NaN: {df['upper_band'].isna().sum()}")
        
        # チャネルブレイクアウトシグナルの検出
        df['upper_breakout'] = (df['close'] > df['upper_band']) & df['upper_band'].notna()
        df['lower_breakout'] = (df['close'] < df['lower_band']) & df['lower_band'].notna()
        
        # チャネル幅の変化（エクスパンション・コントラクション）
        df['bandwidth_change'] = df['bandwidth'].pct_change()
        df['expanding'] = df['bandwidth_change'] > 0.05  # 5%以上の拡大
        df['contracting'] = df['bandwidth_change'] < -0.05  # 5%以上の縮小
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # HyperFRAMAChannelのプロット設定
        main_plots.append(mpf.make_addplot(df['midline'], color='blue', width=2.5, label='FRAMA Midline'))
        main_plots.append(mpf.make_addplot(df['upper_band'], color='red', width=1.5, alpha=0.8, label='Upper Band'))
        main_plots.append(mpf.make_addplot(df['lower_band'], color='green', width=1.5, alpha=0.8, label='Lower Band'))
        
        # チャネルの塗りつぶし（fill_between風の効果）
        # mplfinanceではfill_betweenが直接使えないため、バンドのプロットで代用
        
        # ブレイクアウトシグナルのプロット用データ準備
        upper_breakout_scatter = pd.Series(index=df.index, data=np.nan)
        lower_breakout_scatter = pd.Series(index=df.index, data=np.nan)
        
        # ブレイクアウトがある箇所に価格を設定
        upper_breakout_scatter.loc[df['upper_breakout']] = df.loc[df['upper_breakout'], 'close']
        lower_breakout_scatter.loc[df['lower_breakout']] = df.loc[df['lower_breakout'], 'close']
        
        # シグナルが存在する場合のみプロットに追加
        if upper_breakout_scatter.notna().any():
            main_plots.append(mpf.make_addplot(upper_breakout_scatter, type='scatter', markersize=80, marker='^', color='red', alpha=0.8))
        if lower_breakout_scatter.notna().any():
            main_plots.append(mpf.make_addplot(lower_breakout_scatter, type='scatter', markersize=80, marker='v', color='green', alpha=0.8))
        
        # 2. オシレータープロット
        # ATRとボラティリティレジームパネル
        atr_panel = mpf.make_addplot(df['atr_values'], panel=1, color='orange', width=1.5, 
                                    ylabel='ATR & Vol Regime', secondary_y=False, label='ATR')
        
        # ボラティリティレジームがある場合
        if 'volatility_regime' in df.columns:
            vol_regime_panel = mpf.make_addplot(df['volatility_regime'], panel=1, color='purple', width=1.2, 
                                               secondary_y=True, label='Vol Regime')
        
        # 動的乗数とHyperERパネル
        multiplier_panel = mpf.make_addplot(df['multiplier_values'], panel=2, color='navy', width=1.5, 
                                          ylabel='Multiplier & ER', secondary_y=False, label='Multiplier')
        
        # HyperER値がある場合
        if 'er_values' in df.columns and df['er_values'].notna().any():
            er_panel = mpf.make_addplot(df['er_values'], panel=2, color='darkred', width=1.2, 
                                       secondary_y=True, label='HyperER')
        
        # チャネル幅とパーセンタイルパネル
        bandwidth_panel = mpf.make_addplot(df['bandwidth'], panel=3, color='brown', width=1.5, 
                                         ylabel='Bandwidth & Percentile', secondary_y=False, label='Bandwidth')
        
        # パーセンタイル分析がある場合
        if 'channel_width_percentile' in df.columns:
            percentile_panel = mpf.make_addplot(df['channel_width_percentile'], panel=3, color='teal', width=1.2, 
                                              secondary_y=True, label='Width Percentile')
        
        # チャネルポジションパネル
        position_panel = mpf.make_addplot(df['channel_position'], panel=4, color='magenta', width=2.0, 
                                        ylabel='Channel Position', secondary_y=False, label='Position')
        
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
        
        # 出来高と追加パネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:ATR:乗数:バンド幅:ポジション
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            atr_panel = mpf.make_addplot(df['atr_values'], panel=2, color='orange', width=1.5, 
                                        ylabel='ATR & Vol Regime', secondary_y=False, label='ATR')
            if 'volatility_regime' in df.columns:
                vol_regime_panel = mpf.make_addplot(df['volatility_regime'], panel=2, color='purple', width=1.2, 
                                                   secondary_y=True, label='Vol Regime')
            
            multiplier_panel = mpf.make_addplot(df['multiplier_values'], panel=3, color='navy', width=1.5, 
                                              ylabel='Multiplier & ER', secondary_y=False, label='Multiplier')
            if 'er_values' in df.columns and df['er_values'].notna().any():
                er_panel = mpf.make_addplot(df['er_values'], panel=3, color='darkred', width=1.2, 
                                           secondary_y=True, label='HyperER')
            
            bandwidth_panel = mpf.make_addplot(df['bandwidth'], panel=4, color='brown', width=1.5, 
                                             ylabel='Bandwidth & Percentile', secondary_y=False, label='Bandwidth')
            if 'channel_width_percentile' in df.columns:
                percentile_panel = mpf.make_addplot(df['channel_width_percentile'], panel=4, color='teal', width=1.2, 
                                                  secondary_y=True, label='Width Percentile')
            
            position_panel = mpf.make_addplot(df['channel_position'], panel=5, color='magenta', width=2.0, 
                                            ylabel='Channel Position', secondary_y=False, label='Position')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:ATR:乗数:バンド幅:ポジション
        
        # すべてのプロットを結合
        all_plots = [main_plots[0], main_plots[1], main_plots[2], atr_panel, multiplier_panel, bandwidth_panel, position_panel]
        
        # 条件付きプロットの追加
        if len(main_plots) > 3:  # ブレイクアウトシグナルがある場合
            all_plots.extend(main_plots[3:])
        
        if 'volatility_regime' in df.columns:
            all_plots.append(vol_regime_panel)
        
        if 'er_values' in df.columns and df['er_values'].notna().any():
            all_plots.append(er_panel)
        
        if 'channel_width_percentile' in df.columns:
            all_plots.append(percentile_panel)
        
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['FRAMA Midline', 'Upper Band', 'Lower Band', 'Upper Breakout', 'Lower Breakout'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # ATRパネル
        axes[panel_offset].axhline(y=df['atr_values'].median(), color='orange', linestyle='--', alpha=0.5, label='ATR Median')
        axes[panel_offset].legend(loc='upper right', fontsize=8)
        
        # 乗数パネル
        axes[panel_offset + 1].axhline(y=2.5, color='black', linestyle='--', alpha=0.5, label='Default Multiplier')
        axes[panel_offset + 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
        axes[panel_offset + 1].axhline(y=3.0, color='red', linestyle='--', alpha=0.3)
        axes[panel_offset + 1].legend(loc='upper right', fontsize=8)
        
        # パーセンタイルパネル（右軸がある場合）
        if 'channel_width_percentile' in df.columns:
            axes[panel_offset + 2].axhline(y=50, color='teal', linestyle='--', alpha=0.5, label='50th Percentile')
            axes[panel_offset + 2].axhline(y=25, color='green', linestyle='--', alpha=0.3)
            axes[panel_offset + 2].axhline(y=75, color='red', linestyle='--', alpha=0.3)
        
        # ポジションパネル
        axes[panel_offset + 3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_offset + 3].axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Upper Break')
        axes[panel_offset + 3].axhline(y=-1, color='green', linestyle='--', alpha=0.3, label='Lower Break')
        axes[panel_offset + 3].legend(loc='upper right', fontsize=8)
        
        # 統計情報の表示
        print(f"\n=== HyperFRAMAChannel統計 ===")
        total_points = len(df.dropna())
        upper_breakouts = int(df['upper_breakout'].sum())
        lower_breakouts = int(df['lower_breakout'].sum())
        
        print(f"総データ点数: {total_points}")
        print(f"上限ブレイクアウト: {upper_breakouts}")
        print(f"下限ブレイクアウト: {lower_breakouts}")
        
        # NaN値を除いた統計計算
        atr_clean = df['atr_values'].dropna()
        multiplier_clean = df['multiplier_values'].dropna()
        bandwidth_clean = df['bandwidth'].dropna()
        
        if len(atr_clean) > 0:
            print(f"ATR - 平均: {atr_clean.mean():.4f}, 範囲: {atr_clean.min():.4f} - {atr_clean.max():.4f}")
        if len(multiplier_clean) > 0:
            print(f"乗数 - 平均: {multiplier_clean.mean():.3f}, 範囲: {multiplier_clean.min():.3f} - {multiplier_clean.max():.3f}")
        if len(bandwidth_clean) > 0:
            print(f"チャネル幅 - 平均: {bandwidth_clean.mean():.4f}, 範囲: {bandwidth_clean.min():.4f} - {bandwidth_clean.max():.4f}")
        
        if 'er_values' in df.columns:
            er_clean = df['er_values'].dropna()
            if len(er_clean) > 0:
                print(f"HyperER - 平均: {er_clean.mean():.3f}, 範囲: {er_clean.min():.3f} - {er_clean.max():.3f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='HyperFRAMAChannelの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=14, help='基本期間')
    parser.add_argument('--multiplier-mode', type=str, default='dynamic', help='乗数モード (fixed または dynamic)')
    parser.add_argument('--fixed-multiplier', type=float, default=2.5, help='固定乗数値')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--hyper-frama-period', type=int, default=16, help='HyperFRAMA期間')
    parser.add_argument('--x-atr-period', type=float, default=12.0, help='X_ATR期間')
    parser.add_argument('--hyper-er-period', type=int, default=8, help='HyperER期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperFRAMAChannelChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        multiplier_mode=args.multiplier_mode,
        fixed_multiplier=args.fixed_multiplier,
        src_type=args.src_type,
        hyper_frama_period=args.hyper_frama_period,
        x_atr_period=args.x_atr_period,
        hyper_er_period=args.hyper_er_period
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()