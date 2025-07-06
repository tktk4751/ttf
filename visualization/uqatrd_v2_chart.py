
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
from indicators.ultra_quantum_adaptive_trend_range_discriminator_v2 import UltraQuantumAdaptiveTrendRangeDiscriminator


class UQATRDV2Chart:
    """
    Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD) V2を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - UQATRD V2のトレンド/レンジ判定信号
    - 信号強度
    - ドミナントサイクル周期
    - 適応的閾値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.uqatrd_v2 = None
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
                            # UQATRD V2 DSP Engine Parameters
                            dc_period: int = 40,
                            bandwidth: float = 0.1,
                            dc_smooth_period: int = 10,
                            # General Parameters
                            src_type: str = 'ukf_hlc3',
                            min_data_points: int = 100
                           ) -> None:
        """
        UQATRD V2を計算する
        
        Args:
            dc_period: ドミナントサイクル測定の基準周期 (通常20-60)
            bandwidth: バンドパスフィルターの帯域幅 (通常0.1-0.3)
            dc_smooth_period: ドミナントサイクル平滑化期間（UltimateSmoother使用）
            src_type: 価格ソースタイプ
            min_data_points: 最小データポイント数
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nUQATRD V2を計算中...")
        
        # UQATRD V2を計算
        self.uqatrd_v2 = UltraQuantumAdaptiveTrendRangeDiscriminator(
            dc_period=dc_period,
            bandwidth=bandwidth,
            dc_smooth_period=dc_smooth_period,
            src_type=src_type,
            min_data_points=min_data_points
        )
        
        # UQATRD V2の計算
        print("計算を実行します...")
        result = self.uqatrd_v2.calculate(self.data)
        
        # 結果の取得テスト
        trend_range_signal = self.uqatrd_v2.get_trend_range_signal()
        signal_strength = self.uqatrd_v2.get_signal_strength()
        dominant_cycle = self.uqatrd_v2.get_dominant_cycle_period()
        adaptive_threshold = self.uqatrd_v2.get_adaptive_threshold()
        
        print(f"UQATRD V2計算完了 - 信号: {len(trend_range_signal)}, 強度: {len(signal_strength)}")
        print(f"ドミナントサイクル: {len(dominant_cycle)}, 閾値: {len(adaptive_threshold)}")
        
        # 統計情報の表示
        if trend_range_signal is not None:
            trend_count = (trend_range_signal > 0.5).sum()
            range_count = (trend_range_signal <= 0.5).sum()
            total_count = len(trend_range_signal)
            
            print(f"トレンド判定 - トレンド: {trend_count} ({trend_count/total_count*100:.1f}%)")
            print(f"レンジ判定 - レンジ: {range_count} ({range_count/total_count*100:.1f}%)")
            print(f"信号強度 - 平均: {signal_strength.mean():.3f}, 範囲: {signal_strength.min():.3f} - {signal_strength.max():.3f}")
            print(f"ドミナントサイクル - 平均: {dominant_cycle.mean():.1f}, 範囲: {dominant_cycle.min():.1f} - {dominant_cycle.max():.1f}")
        
        print("UQATRD V2計算完了")
            
    def plot(self, 
            title: str = "Ultra Quantum Adaptive Trend-Range Discriminator V2", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUQATRD V2を描画する
        
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
            
        if self.uqatrd_v2 is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # UQATRD V2の値を取得
        print("UQATRD V2データを取得中...")
        trend_range_signal = self.uqatrd_v2.get_trend_range_signal()
        signal_strength = self.uqatrd_v2.get_signal_strength()
        dominant_cycle = self.uqatrd_v2.get_dominant_cycle_period()
        adaptive_threshold = self.uqatrd_v2.get_adaptive_threshold()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_range_signal': trend_range_signal,
                'signal_strength': signal_strength,
                'dominant_cycle': dominant_cycle,
                'adaptive_threshold': adaptive_threshold
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"UQATRD V2データ確認 - 信号NaN: {df['trend_range_signal'].isna().sum()}, 強度NaN: {df['signal_strength'].isna().sum()}")
        
        # トレンド/レンジ判定の色分け
        df['trend_signal'] = np.where(df['trend_range_signal'] > 0.5, df['trend_range_signal'], np.nan)
        df['range_signal'] = np.where(df['trend_range_signal'] <= 0.5, df['trend_range_signal'], np.nan)
        
        # トレンド/レンジ分類（0=レンジ, 1=トレンド）
        df['trend_range_classification'] = (df['trend_range_signal'] > 0.5).astype(float)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['trend_range_signal'].isna() | df['signal_strength'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # UQATRD V2のプロット設定
        main_plots.append(mpf.make_addplot(df['trend_signal'], color='green', width=2, alpha=0.8, label='Trend Signal'))
        main_plots.append(mpf.make_addplot(df['range_signal'], color='red', width=2, alpha=0.8, label='Range Signal'))
        
        # 2. オシレータープロット
        # トレンド/レンジ信号パネル
        signal_panel = mpf.make_addplot(df['trend_range_signal'], panel=1, color='blue', width=1.5, 
                                       ylabel='Trend/Range Signal', secondary_y=False, label='Signal')
        
        # 信号強度パネル
        strength_panel = mpf.make_addplot(df['signal_strength'], panel=2, color='purple', width=1.5, 
                                         ylabel='Signal Strength', secondary_y=False, label='Strength')
        
        # ドミナントサイクルパネル
        cycle_panel = mpf.make_addplot(df['dominant_cycle'], panel=3, color='orange', width=1.5, 
                                      ylabel='Dominant Cycle', secondary_y=False, label='Cycle')
        
        # トレンド/レンジ分類パネル
        classification_panel = mpf.make_addplot(df['trend_range_classification'], panel=4, color='brown', width=2, 
                                               ylabel='Classification', secondary_y=False, label='Class', type='line')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:信号:強度:サイクル:分類
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            signal_panel = mpf.make_addplot(df['trend_range_signal'], panel=2, color='blue', width=1.5, 
                                           ylabel='Trend/Range Signal', secondary_y=False, label='Signal')
            strength_panel = mpf.make_addplot(df['signal_strength'], panel=3, color='purple', width=1.5, 
                                             ylabel='Signal Strength', secondary_y=False, label='Strength')
            cycle_panel = mpf.make_addplot(df['dominant_cycle'], panel=4, color='orange', width=1.5, 
                                          ylabel='Dominant Cycle', secondary_y=False, label='Cycle')
            classification_panel = mpf.make_addplot(df['trend_range_classification'], panel=5, color='brown', width=2, 
                                                   ylabel='Classification', secondary_y=False, label='Class', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:信号:強度:サイクル:分類
        
        # すべてのプロットを結合
        all_plots = main_plots + [signal_panel, strength_panel, cycle_panel, classification_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Trend Signal', 'Range Signal'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # トレンド/レンジ信号パネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # 信号強度パネル
            axes[3].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # ドミナントサイクルパネル
            cycle_mean = df['dominant_cycle'].mean()
            axes[4].axhline(y=cycle_mean, color='black', linestyle='-', alpha=0.3)
            
            # 分類パネル
            axes[5].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Range')
            axes[5].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Trend')
        else:
            # トレンド/レンジ信号パネル
            axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # 信号強度パネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # ドミナントサイクルパネル
            cycle_mean = df['dominant_cycle'].mean()
            axes[3].axhline(y=cycle_mean, color='black', linestyle='-', alpha=0.3)
            
            # 分類パネル
            axes[4].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Range')
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Trend')
        
        # 統計情報の表示
        print(f"\n=== UQATRD V2統計 ===")
        total_points = len(df[df['trend_range_signal'].notna()])
        trend_points = len(df[df['trend_range_classification'] == 1])
        range_points = len(df[df['trend_range_classification'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"トレンド判定: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"レンジ判定: {range_points} ({range_points/total_points*100:.1f}%)")
        print(f"信号強度 - 平均: {df['signal_strength'].mean():.3f}, 範囲: {df['signal_strength'].min():.3f} - {df['signal_strength'].max():.3f}")
        print(f"ドミナントサイクル - 平均: {df['dominant_cycle'].mean():.1f}, 範囲: {df['dominant_cycle'].min():.1f} - {df['dominant_cycle'].max():.1f}")
        
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
    parser = argparse.ArgumentParser(description='UQATRD V2の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='ukf_hlc3', help='価格ソースタイプ')
    parser.add_argument('--dc-period', type=int, default=40, help='ドミナントサイクル測定の基準周期')
    parser.add_argument('--bandwidth', type=float, default=0.1, help='バンドパスフィルターの帯域幅')
    parser.add_argument('--dc-smooth-period', type=int, default=10, help='ドミナントサイクル平滑化期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UQATRDV2Chart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        dc_period=args.dc_period,
        bandwidth=args.bandwidth,
        dc_smooth_period=args.dc_smooth_period,
        src_type=args.src_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 