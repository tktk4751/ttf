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
from indicators.smoother.roofing_filter import RoofingFilter, SuperSmoother, HighPassFilter


class RoofingFilterChart:
    """
    Roofingフィルターを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Roofingフィルター（メインフィルター値）
    - 元の価格データ vs フィルター済みデータ
    - HighPassフィルター値（別パネル）
    - SuperSmootherフィルター値（別パネル）
    - スペクトラル・ディラテーション除去の効果を可視化
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.roofing_filter = None
        self.supersmoother = None
        self.highpass = None
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
                            # Roofingフィルターパラメータ
                            src_type: str = 'close',
                            hp_cutoff: float = 48.0,
                            ss_band_edge: float = 10.0,
                            # 比較用個別フィルターパラメータ
                            supersmoother_band_edge: float = 10.0,
                            highpass_cutoff: float = 48.0
                           ) -> None:
        """
        Roofingフィルターと関連するフィルターを計算する
        
        Args:
            src_type: ソースタイプ
            hp_cutoff: HighPassフィルターのカットオフ周期（デフォルト: 48）
            ss_band_edge: SuperSmootherフィルターのバンドエッジ周期（デフォルト: 10）
            supersmoother_band_edge: 比較用SuperSmootherのバンドエッジ周期
            highpass_cutoff: 比較用HighPassのカットオフ周期
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nRoofingフィルターを計算中...")
        
        # Roofingフィルター（メイン）
        self.roofing_filter = RoofingFilter(
            src_type=src_type,
            hp_cutoff=hp_cutoff,
            ss_band_edge=ss_band_edge
        )
        
        # 比較用個別フィルター
        self.supersmoother = SuperSmoother(
            band_edge=supersmoother_band_edge,
            src_type=src_type
        )
        
        self.highpass = HighPassFilter(
            cutoff_period=highpass_cutoff,
            src_type=src_type
        )
        
        print("計算を実行します...")
        
        # 各フィルターの計算
        roofing_result = self.roofing_filter.calculate(self.data)
        supersmoother_values = self.supersmoother.calculate(self.data)
        highpass_values = self.highpass.calculate(self.data)
        
        print(f"Roofingフィルター計算完了")
        print(f"  - Roofingフィルター値: {len(roofing_result.values)}")
        print(f"  - HighPass値: {len(roofing_result.highpass)}")
        print(f"  - SuperSmoother値: {len(roofing_result.supersmoother)}")
        print(f"個別フィルター計算完了")
        print(f"  - SuperSmoother単体: {len(supersmoother_values)}")
        print(f"  - HighPass単体: {len(highpass_values)}")
        
        # NaN値のチェック
        roofing_nan = np.isnan(roofing_result.values).sum()
        hp_nan = np.isnan(roofing_result.highpass).sum()
        ss_nan = np.isnan(roofing_result.supersmoother).sum()
        
        print(f"NaN値チェック:")
        print(f"  - Roofing: {roofing_nan}, HighPass: {hp_nan}, SuperSmoother: {ss_nan}")
        
        # 結果を保存
        self._roofing_result = roofing_result
        self._supersmoother_values = supersmoother_values
        self._highpass_values = highpass_values
        
        print("全フィルター計算完了")
            
    def plot(self, 
            title: str = "Roofingフィルター分析", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとRoofingフィルターを描画する
        
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
            
        if self.roofing_filter is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # フィルター結果の取得
        print("フィルターデータを取得中...")
        roofing_values = self._roofing_result.values
        highpass_values = self._roofing_result.highpass
        supersmoother_values = self._roofing_result.supersmoother
        
        # 比較用の個別フィルター値
        standalone_ss = self._supersmoother_values
        standalone_hp = self._highpass_values
        
        # 元の価格データ（比較用）
        from indicators.price_source import PriceSource
        original_price = PriceSource.calculate_source(self.data, self.roofing_filter.src_type)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'original_price': original_price,
                'roofing_filter': roofing_values,
                'roofing_highpass': highpass_values,
                'roofing_supersmoother': supersmoother_values,
                'standalone_supersmoother': standalone_ss,
                'standalone_highpass': standalone_hp
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"フィルターデータ確認 - RoofingNaN: {df['roofing_filter'].isna().sum()}")
        
        # データの詳細確認
        print(f"データ範囲確認:")
        print(f"  - 元価格: {df['original_price'].min():.4f} - {df['original_price'].max():.4f}")
        print(f"  - Roofingフィルター: {df['roofing_filter'].min():.4f} - {df['roofing_filter'].max():.4f}")
        print(f"  - 有効データ数（元価格）: {df['original_price'].count()}")
        print(f"  - 有効データ数（Roofing）: {df['roofing_filter'].count()}")
        
        # NaN以外のデータが存在するか確認
        if df['roofing_filter'].count() == 0:
            print("警告: Roofingフィルターに有効なデータがありません！")
            return
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 1. メインチャート上のプロット（panelを指定しない＝メインチャート）
        # 元の価格と各フィルター結果を重ねて表示
        main_plots.append(mpf.make_addplot(df['original_price'], color='gray', width=1, alpha=0.7))
        main_plots.append(mpf.make_addplot(df['roofing_filter'], color='blue', width=2))
        
        # 2. HighPassフィルターパネル
        # Roofing内のHighPassと単体HighPassを比較
        highpass_panel1 = mpf.make_addplot(df['roofing_highpass'], panel=1, color='red', width=1.5, 
                                          ylabel='HighPass Filter', secondary_y=False, label='Roofing HighPass')
        highpass_panel2 = mpf.make_addplot(df['standalone_highpass'], panel=1, color='orange', width=1, alpha=0.7,
                                          secondary_y=False, label='Standalone HighPass')
        
        # 3. SuperSmootherフィルターパネル
        # Roofing内のSuperSmootherと単体SuperSmootherを比較
        supersmoother_panel1 = mpf.make_addplot(df['roofing_supersmoother'], panel=2, color='green', width=1.5, 
                                               ylabel='SuperSmoother Filter', secondary_y=False, label='Roofing SuperSmoother')
        supersmoother_panel2 = mpf.make_addplot(df['standalone_supersmoother'], panel=2, color='lime', width=1, alpha=0.7,
                                               secondary_y=False, label='Standalone SuperSmoother')
        
        # 4. フィルター効果比較パネル
        # 元価格からの偏差を表示してフィルタリング効果を可視化
        df['price_deviation'] = df['original_price'] - df['original_price'].rolling(window=20).mean()
        df['roofing_deviation'] = df['roofing_filter'] - df['roofing_filter'].rolling(window=20).mean()
        
        deviation_panel1 = mpf.make_addplot(df['price_deviation'], panel=3, color='gray', width=1, 
                                           ylabel='Deviation from MA20', secondary_y=False, label='Original Deviation')
        deviation_panel2 = mpf.make_addplot(df['roofing_deviation'], panel=3, color='blue', width=1.5,
                                           secondary_y=False, label='Roofing Deviation')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:HighPass:SuperSmoother:偏差比較
            # 出来高を表示する場合は、パネル番号を+1する
            highpass_panel1 = mpf.make_addplot(df['roofing_highpass'], panel=2, color='red', width=1.5, 
                                              ylabel='HighPass Filter', secondary_y=False, label='Roofing HighPass')
            highpass_panel2 = mpf.make_addplot(df['standalone_highpass'], panel=2, color='orange', width=1, alpha=0.7,
                                              secondary_y=False, label='Standalone HighPass')
            supersmoother_panel1 = mpf.make_addplot(df['roofing_supersmoother'], panel=3, color='green', width=1.5, 
                                                   ylabel='SuperSmoother Filter', secondary_y=False, label='Roofing SuperSmoother')
            supersmoother_panel2 = mpf.make_addplot(df['standalone_supersmoother'], panel=3, color='lime', width=1, alpha=0.7,
                                                   secondary_y=False, label='Standalone SuperSmoother')
            deviation_panel1 = mpf.make_addplot(df['price_deviation'], panel=4, color='gray', width=1, 
                                               ylabel='Deviation from MA20', secondary_y=False, label='Original Deviation')
            deviation_panel2 = mpf.make_addplot(df['roofing_deviation'], panel=4, color='blue', width=1.5,
                                               secondary_y=False, label='Roofing Deviation')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:HighPass:SuperSmoother:偏差比較
        
        # すべてのプロットを結合
        all_plots = main_plots + [highpass_panel1, highpass_panel2, supersmoother_panel1, supersmoother_panel2, 
                                 deviation_panel1, deviation_panel2]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（メインチャート）
        # mplfinanceの仕様上、addplotの凡例は自動では表示されないため、手動で追加
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', alpha=0.7, linewidth=1, label='Original Price'),
            Line2D([0], [0], color='blue', linewidth=2, label='Roofing Filter')
        ]
        axes[0].legend(handles=legend_elements, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # HighPassフィルターパネル
        hp_panel_idx = 1 + panel_offset
        axes[hp_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[hp_panel_idx].legend(['Roofing HighPass', 'Standalone HighPass'], loc='upper right', fontsize=8)
        
        # SuperSmootherフィルターパネル
        ss_panel_idx = 2 + panel_offset
        axes[ss_panel_idx].legend(['Roofing SuperSmoother', 'Standalone SuperSmoother'], loc='upper right', fontsize=8)
        
        # 偏差比較パネル
        dev_panel_idx = 3 + panel_offset
        axes[dev_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[dev_panel_idx].legend(['Original Deviation', 'Roofing Deviation'], loc='upper right', fontsize=8)
        
        # 統計情報の表示
        print(f"\n=== Roofingフィルター統計 ===")
        print(f"データポイント数: {len(df)}")
        
        # フィルタリング効果の統計
        original_std = df['original_price'].std()
        roofing_std = df['roofing_filter'].std()
        reduction_ratio = (original_std - roofing_std) / original_std * 100
        
        print(f"価格ボラティリティ（標準偏差）:")
        print(f"  - 元データ: {original_std:.4f}")
        print(f"  - Roofingフィルター後: {roofing_std:.4f}")
        print(f"  - ノイズ削減率: {reduction_ratio:.1f}%")
        
        # HighPassとSuperSmootherの範囲
        hp_range = df['roofing_highpass'].max() - df['roofing_highpass'].min()
        ss_range = df['roofing_supersmoother'].max() - df['roofing_supersmoother'].min()
        print(f"フィルター範囲:")
        print(f"  - HighPass: {hp_range:.4f}")
        print(f"  - SuperSmoother: {ss_range:.4f}")
        
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
    parser = argparse.ArgumentParser(description='Roofingフィルターの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='close', help='ソースタイプ')
    parser.add_argument('--hp-cutoff', type=float, default=48.0, help='HighPassカットオフ周期')
    parser.add_argument('--ss-band-edge', type=float, default=10.0, help='SuperSmootherバンドエッジ周期')
    args = parser.parse_args()
    
    # チャートを作成
    chart = RoofingFilterChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        src_type=args.src_type,
        hp_cutoff=args.hp_cutoff,
        ss_band_edge=args.ss_band_edge
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()