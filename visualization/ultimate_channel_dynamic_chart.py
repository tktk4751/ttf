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
from indicators.ultimate_channel import UltimateChannel


class UltimateChannelDynamicChart:
    """
    Ultimate Channelの動的乗数機能を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 固定乗数と動的乗数のUltimate Channel
    - UQATRD信号値とその閾値
    - 動的乗数の変化
    - チャネル幅の比較
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.fixed_channel = None
        self.dynamic_channel = None
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
                            # Ultimate Channel パラメータ
                            length: float = 20.0,
                            str_length: float = 20.0,
                            fixed_num_strs: float = 2.0,
                            src_type: str = 'hlc3',
                            midband_type: str = 'ultimate_smoother',
                            
                            # UQATRD パラメータ（動的モード用）
                            uqatrd_coherence_window: int = 21,
                            uqatrd_entanglement_window: int = 34,
                            uqatrd_efficiency_window: int = 21,
                            uqatrd_uncertainty_window: int = 14,
                            uqatrd_str_period: float = 20.0
                           ) -> None:
        """
        固定乗数と動的乗数のUltimate Channelを計算する
        
        Args:
            length: 中心線の期間
            str_length: STR期間
            fixed_num_strs: 固定乗数
            src_type: プライスソースタイプ
            midband_type: ミッドバンドタイプ
            uqatrd_coherence_window: UQATRD量子コヒーレンス分析窓
            uqatrd_entanglement_window: UQATRD量子エンタングルメント分析窓
            uqatrd_efficiency_window: UQATRD量子効率スペクトラム分析窓
            uqatrd_uncertainty_window: UQATRD量子不確定性分析窓
            uqatrd_str_period: UQATRD用STR期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nUltimate Channelを計算中...")
        
        # 固定乗数チャネル
        print("固定乗数チャネルを計算中...")
        self.fixed_channel = UltimateChannel(
            length=length,
            str_length=str_length,
            num_strs=fixed_num_strs,
            src_type=src_type,
            midband_type=midband_type,
            multiplier_mode='fixed'
        )
        
        # 動的乗数チャネル
        print("動的乗数チャネルを計算中...")
        self.dynamic_channel = UltimateChannel(
            length=length,
            str_length=str_length,
            num_strs=fixed_num_strs,  # 基準値（使用されない）
            src_type=src_type,
            midband_type=midband_type,
            multiplier_mode='dynamic',
            uqatrd_coherence_window=uqatrd_coherence_window,
            uqatrd_entanglement_window=uqatrd_entanglement_window,
            uqatrd_efficiency_window=uqatrd_efficiency_window,
            uqatrd_uncertainty_window=uqatrd_uncertainty_window,
            uqatrd_str_period=uqatrd_str_period
        )
        
        # Ultimate Channelの計算
        print("チャネル計算を実行中...")
        fixed_result = self.fixed_channel.calculate(self.data)
        dynamic_result = self.dynamic_channel.calculate(self.data)
        
        print("Ultimate Channel計算完了")
        
        # 結果の検証
        print(f"固定チャネル - 上限: {len(fixed_result.upper_channel)}, 下限: {len(fixed_result.lower_channel)}")
        print(f"動的チャネル - 上限: {len(dynamic_result.upper_channel)}, 下限: {len(dynamic_result.lower_channel)}")
        
        # 動的乗数の統計
        dynamic_multipliers = self.dynamic_channel.get_dynamic_multipliers()
        uqatrd_values = self.dynamic_channel.get_uqatrd_values()
        
        if dynamic_multipliers is not None:
            print(f"動的乗数 - 平均: {np.mean(dynamic_multipliers):.2f}, 範囲: {np.min(dynamic_multipliers):.1f} - {np.max(dynamic_multipliers):.1f}")
        
        if uqatrd_values is not None:
            print(f"UQATRD信号 - 平均: {np.mean(uqatrd_values):.3f}, 範囲: {np.min(uqatrd_values):.3f} - {np.max(uqatrd_values):.3f}")
            
    def plot(self, 
            title: str = "Ultimate Channel - 動的乗数 vs 固定乗数", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUltimate Channelを描画する
        
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
            
        if self.fixed_channel is None or self.dynamic_channel is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Ultimate Channelの値を取得
        print("チャネルデータを取得中...")
        
        # 固定チャネル
        fixed_result = self.fixed_channel.calculate(self.data)
        fixed_upper = fixed_result.upper_channel
        fixed_lower = fixed_result.lower_channel
        fixed_center = fixed_result.center_line
        
        # 動的チャネル
        dynamic_result = self.dynamic_channel.calculate(self.data)
        dynamic_upper = dynamic_result.upper_channel
        dynamic_lower = dynamic_result.lower_channel
        dynamic_center = dynamic_result.center_line
        
        # 追加データ
        dynamic_multipliers = self.dynamic_channel.get_dynamic_multipliers()
        uqatrd_values = self.dynamic_channel.get_uqatrd_values()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'fixed_upper': fixed_upper,
                'fixed_lower': fixed_lower,
                'fixed_center': fixed_center,
                'dynamic_upper': dynamic_upper,
                'dynamic_lower': dynamic_lower,
                'dynamic_center': dynamic_center,
                'dynamic_multipliers': dynamic_multipliers,
                'uqatrd_values': uqatrd_values,
                'fixed_width': fixed_upper - fixed_lower,
                'dynamic_width': dynamic_upper - dynamic_lower
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # 固定チャネル（赤色系）
        main_plots.append(mpf.make_addplot(df['fixed_upper'], color='red', width=1.5, alpha=0.7, label='Fixed Upper'))
        main_plots.append(mpf.make_addplot(df['fixed_lower'], color='red', width=1.5, alpha=0.7, label='Fixed Lower'))
        main_plots.append(mpf.make_addplot(df['fixed_center'], color='darkred', width=1, linestyle='--', alpha=0.8, label='Fixed Center'))
        
        # 動的チャネル（緑色系）
        main_plots.append(mpf.make_addplot(df['dynamic_upper'], color='green', width=1.5, alpha=0.7, label='Dynamic Upper'))
        main_plots.append(mpf.make_addplot(df['dynamic_lower'], color='green', width=1.5, alpha=0.7, label='Dynamic Lower'))
        main_plots.append(mpf.make_addplot(df['dynamic_center'], color='darkgreen', width=1, linestyle='--', alpha=0.8, label='Dynamic Center'))
        
        # 2. 追加パネルのプロット
        # UQATRD信号値パネル
        uqatrd_panel = mpf.make_addplot(df['uqatrd_values'], panel=1, color='purple', width=1.2, 
                                       ylabel='UQATRD Signal', secondary_y=False, label='UQATRD')
        
        # 動的乗数パネル
        mult_panel = mpf.make_addplot(df['dynamic_multipliers'], panel=2, color='blue', width=1.2, 
                                     ylabel='Dynamic Multiplier', secondary_y=False, label='Multiplier')
        
        # チャネル幅比較パネル
        width_fixed_panel = mpf.make_addplot(df['fixed_width'], panel=3, color='red', width=1.2, 
                                           ylabel='Channel Width', secondary_y=False, label='Fixed Width')
        width_dynamic_panel = mpf.make_addplot(df['dynamic_width'], panel=3, color='green', width=1.2, 
                                             secondary_y=False, label='Dynamic Width')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:UQATRD:乗数:幅
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            uqatrd_panel = mpf.make_addplot(df['uqatrd_values'], panel=2, color='purple', width=1.2, 
                                           ylabel='UQATRD Signal', secondary_y=False, label='UQATRD')
            mult_panel = mpf.make_addplot(df['dynamic_multipliers'], panel=3, color='blue', width=1.2, 
                                         ylabel='Dynamic Multiplier', secondary_y=False, label='Multiplier')
            width_fixed_panel = mpf.make_addplot(df['fixed_width'], panel=4, color='red', width=1.2, 
                                               ylabel='Channel Width', secondary_y=False, label='Fixed Width')
            width_dynamic_panel = mpf.make_addplot(df['dynamic_width'], panel=4, color='green', width=1.2, 
                                                 secondary_y=False, label='Dynamic Width')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:UQATRD:乗数:幅
        
        # すべてのプロットを結合
        all_plots = main_plots + [uqatrd_panel, mult_panel, width_fixed_panel, width_dynamic_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Fixed Upper', 'Fixed Lower', 'Fixed Center', 'Dynamic Upper', 'Dynamic Lower', 'Dynamic Center'], 
                      loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # UQATRD信号値パネル（UQATRD閾値）
        uqatrd_axis = axes[1 + panel_offset]
        uqatrd_axis.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='0.4 (Mult=6)')
        uqatrd_axis.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 (Mult=4)')
        uqatrd_axis.axhline(y=0.6, color='yellow', linestyle='--', alpha=0.7, label='0.6 (Mult=3)')
        uqatrd_axis.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='0.7 (Mult=2)')
        uqatrd_axis.set_ylim(0, 1)
        
        # 動的乗数パネル
        mult_axis = axes[2 + panel_offset]
        mult_axis.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, label='Fixed Multiplier')
        for mult_val in [1, 2, 3, 4, 6]:
            mult_axis.axhline(y=mult_val, color='gray', linestyle=':', alpha=0.3)
        
        # チャネル幅パネル
        width_axis = axes[3 + panel_offset]
        width_axis.legend(['Fixed Width', 'Dynamic Width'], loc='upper right', fontsize=8)
        
        # 統計情報の表示
        print(f"\n=== Ultimate Channel 統計 ===")
        
        # 有効なデータポイント数
        valid_data = df.dropna()
        total_points = len(valid_data)
        
        print(f"総データ点数: {total_points}")
        
        # UQATRD統計
        if 'uqatrd_values' in valid_data.columns:
            uqatrd_mean = valid_data['uqatrd_values'].mean()
            uqatrd_std = valid_data['uqatrd_values'].std()
            print(f"UQATRD信号 - 平均: {uqatrd_mean:.3f}, 標準偏差: {uqatrd_std:.3f}")
        
        # 動的乗数統計
        if 'dynamic_multipliers' in valid_data.columns:
            mult_mean = valid_data['dynamic_multipliers'].mean()
            mult_std = valid_data['dynamic_multipliers'].std()
            print(f"動的乗数 - 平均: {mult_mean:.2f}, 標準偏差: {mult_std:.2f}")
            
            # 乗数分布
            print("動的乗数分布:")
            for mult_val in [1, 2, 3, 4, 6]:
                count = (valid_data['dynamic_multipliers'] == mult_val).sum()
                percentage = count / total_points * 100 if total_points > 0 else 0
                print(f"  乗数{mult_val}: {count}点 ({percentage:.1f}%)")
        
        # チャネル幅比較
        if 'fixed_width' in valid_data.columns and 'dynamic_width' in valid_data.columns:
            fixed_width_mean = valid_data['fixed_width'].mean()
            dynamic_width_mean = valid_data['dynamic_width'].mean()
            width_ratio = dynamic_width_mean / fixed_width_mean if fixed_width_mean > 0 else 0
            print(f"チャネル幅 - 固定: {fixed_width_mean:.2f}, 動的: {dynamic_width_mean:.2f} (比率: {width_ratio:.2f})")
        
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
    parser = argparse.ArgumentParser(description='Ultimate Channel動的乗数機能の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='プライスソースタイプ')
    parser.add_argument('--midband-type', type=str, default='ultimate_smoother', help='ミッドバンドタイプ')
    parser.add_argument('--length', type=float, default=20.0, help='中心線の期間')
    parser.add_argument('--str-length', type=float, default=20.0, help='STR期間')
    parser.add_argument('--fixed-mult', type=float, default=2.0, help='固定乗数')
    parser.add_argument('--no-volume', action='store_true', help='出来高を表示しない')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateChannelDynamicChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        length=args.length,
        str_length=args.str_length,
        fixed_num_strs=args.fixed_mult,
        src_type=args.src_type,
        midband_type=args.midband_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 