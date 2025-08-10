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
from indicators.quantum_adaptive_flow_channel import QuantumAdaptiveFlowChannel, QAFCResult


class QAFCChart:
    """
    Quantum Adaptive Flow Channel (QAFC) を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - QAFCの中心線・上限チャネル・下限チャネル
    - トレンド方向のカラー表示
    - トレンド強度、信頼度スコア、モメンタムフローの追加パネル
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.qafc = None
        self.qafc_result = None
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
                            # フィルターパラメータ
                            process_noise: float = 0.01,
                            measurement_noise: float = 0.1,
                            # 適応パラメータ
                            noise_window: int = 20,
                            prediction_lookback: int = 10,
                            # チャネルパラメータ
                            base_multiplier: float = 2.0,
                            # データソース
                            src_type: str = 'hlc3'
                           ) -> None:
        """
        QAFCを計算する
        
        Args:
            process_noise: カルマンフィルターのプロセスノイズ
            measurement_noise: カルマンフィルターの観測ノイズ
            noise_window: ノイズ分析ウィンドウサイズ
            prediction_lookback: 予測に使用する期間
            base_multiplier: 基本チャネル幅倍率
            src_type: 価格ソースタイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nQuantum Adaptive Flow Channelを計算中...")
        
        # QAFCを計算
        self.qafc = QuantumAdaptiveFlowChannel(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            noise_window=noise_window,
            prediction_lookback=prediction_lookback,
            base_multiplier=base_multiplier,
            src_type=src_type
        )
        
        # QAFC計算
        print("計算を実行します...")
        self.qafc_result = self.qafc.calculate(self.data)
        
        # 結果の確認
        print(f"チャネル計算完了 - 中心線: {len(self.qafc_result.centerline)}")
        print(f"                   上限: {len(self.qafc_result.upper_channel)}")
        print(f"                   下限: {len(self.qafc_result.lower_channel)}")
        
        # NaN値のチェック
        nan_count_center = np.isnan(self.qafc_result.centerline).sum()
        nan_count_upper = np.isnan(self.qafc_result.upper_channel).sum()
        nan_count_lower = np.isnan(self.qafc_result.lower_channel).sum()
        print(f"NaN値 - 中心線: {nan_count_center}, 上限: {nan_count_upper}, 下限: {nan_count_lower}")
        
        # トレンド統計
        trend_up = (self.qafc_result.trend_direction > 0).sum()
        trend_down = (self.qafc_result.trend_direction < 0).sum()
        trend_neutral = (self.qafc_result.trend_direction == 0).sum()
        print(f"トレンド方向 - 上昇: {trend_up}, 下降: {trend_down}, 中立: {trend_neutral}")
        
        print("QAFC計算完了")
            
    def plot(self, 
            title: str = "Quantum Adaptive Flow Channel (QAFC)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとQAFCを描画する
        
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
            
        if self.qafc_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # QAFC結果をデータフレームに結合
        print("チャネルデータを準備中...")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'centerline': self.qafc_result.centerline,
                'upper_channel': self.qafc_result.upper_channel,
                'lower_channel': self.qafc_result.lower_channel,
                'trend_direction': self.qafc_result.trend_direction,
                'trend_strength': self.qafc_result.trend_strength,
                'confidence_score': self.qafc_result.confidence_score,
                'momentum_flow': self.qafc_result.momentum_flow,
                'volatility_ratio': self.qafc_result.volatility_ratio,
                'noise_level': self.qafc_result.noise_level
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド方向に基づく中心線の色分け
        df['center_uptrend'] = np.where(df['trend_direction'] > 0, df['centerline'], np.nan)
        df['center_downtrend'] = np.where(df['trend_direction'] < 0, df['centerline'], np.nan)
        df['center_neutral'] = np.where(df['trend_direction'] == 0, df['centerline'], np.nan)
        
        # チャネルの色分け（トレンド方向に基づく）
        df['upper_uptrend'] = np.where(df['trend_direction'] > 0, df['upper_channel'], np.nan)
        df['upper_downtrend'] = np.where(df['trend_direction'] < 0, df['upper_channel'], np.nan)
        df['lower_uptrend'] = np.where(df['trend_direction'] > 0, df['lower_channel'], np.nan)
        df['lower_downtrend'] = np.where(df['trend_direction'] < 0, df['lower_channel'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # QAFCチャネルのプロット設定
        # 中心線（NaN値をチェックして有効なデータのみプロット）
        if not df['center_uptrend'].isna().all():
            main_plots.append(mpf.make_addplot(df['center_uptrend'], color='green', width=2, label='QAFC Center (Up)'))
        if not df['center_downtrend'].isna().all():
            main_plots.append(mpf.make_addplot(df['center_downtrend'], color='red', width=2, label='QAFC Center (Down)'))
        if not df['center_neutral'].isna().all():
            main_plots.append(mpf.make_addplot(df['center_neutral'], color='gray', width=2, label='QAFC Center (Neutral)'))
        
        # 上限チャネル
        if not df['upper_uptrend'].isna().all():
            main_plots.append(mpf.make_addplot(df['upper_uptrend'], color='green', width=1, alpha=0.7, linestyle='--', label='Upper (Up)'))
        if not df['upper_downtrend'].isna().all():
            main_plots.append(mpf.make_addplot(df['upper_downtrend'], color='red', width=1, alpha=0.7, linestyle='--', label='Upper (Down)'))
        
        # 下限チャネル
        if not df['lower_uptrend'].isna().all():
            main_plots.append(mpf.make_addplot(df['lower_uptrend'], color='green', width=1, alpha=0.7, linestyle='--', label='Lower (Up)'))
        if not df['lower_downtrend'].isna().all():
            main_plots.append(mpf.make_addplot(df['lower_downtrend'], color='red', width=1, alpha=0.7, linestyle='--', label='Lower (Down)'))
        
        # 2. 追加パネルのプロット
        # パネル番号の初期化
        panel_num = 1 if show_volume else 0
        
        additional_plots = []
        
        # トレンド強度パネル（NaN値チェック）
        if not df['trend_strength'].isna().all():
            trend_strength_panel = mpf.make_addplot(df['trend_strength'], panel=panel_num + 1, color='blue', width=1.2, 
                                                   ylabel='Trend Strength', secondary_y=False, label='Trend Strength')
            additional_plots.append(trend_strength_panel)
        
        # 信頼度スコアパネル（NaN値チェック）
        if not df['confidence_score'].isna().all():
            confidence_panel = mpf.make_addplot(df['confidence_score'], panel=panel_num + 2, color='purple', width=1.2, 
                                              ylabel='Confidence Score', secondary_y=False, label='Confidence')
            additional_plots.append(confidence_panel)
        
        # モメンタムフローパネル（NaN値チェック）
        if not df['momentum_flow'].isna().all():
            momentum_panel = mpf.make_addplot(df['momentum_flow'], panel=panel_num + 3, color='orange', width=1.2, 
                                            ylabel='Momentum Flow', secondary_y=False, label='Momentum', type='bar')
            additional_plots.append(momentum_panel)
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:強度:信頼度:モメンタム
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:強度:信頼度:モメンタム
        
        # すべてのプロットを結合
        all_plots = main_plots + additional_plots
        if all_plots:  # プロットが存在する場合のみ追加
            kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（メインチャート）
        axes[0].legend(['QAFC Center (Up)', 'QAFC Center (Down)', 'Upper Channel', 'Lower Channel'], 
                      loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # トレンド強度パネル（0-1の範囲）
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_ylim(-0.1, 1.1)
            
            # 信頼度スコアパネル（0-1の範囲）
            axes[3].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[3].set_ylim(-0.1, 1.1)
            
            # モメンタムフローパネル（中心線を0に）
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            # トレンド強度パネル（0-1の範囲）
            axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_ylim(-0.1, 1.1)
            
            # 信頼度スコアパネル（0-1の範囲）
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_ylim(-0.1, 1.1)
            
            # モメンタムフローパネル（中心線を0に）
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== QAFC統計情報 ===")
        valid_data = df.dropna(subset=['trend_direction'])
        if len(valid_data) > 0:
            print(f"総データ点数: {len(valid_data)}")
            
            # トレンド統計
            trend_up = (valid_data['trend_direction'] > 0).sum()
            trend_down = (valid_data['trend_direction'] < 0).sum()
            trend_neutral = (valid_data['trend_direction'] == 0).sum()
            
            print(f"上昇トレンド: {trend_up} ({trend_up/len(valid_data)*100:.1f}%)")
            print(f"下降トレンド: {trend_down} ({trend_down/len(valid_data)*100:.1f}%)")
            print(f"中立: {trend_neutral} ({trend_neutral/len(valid_data)*100:.1f}%)")
            
            # その他の統計
            print(f"\n平均トレンド強度: {valid_data['trend_strength'].mean():.3f}")
            print(f"平均信頼度スコア: {valid_data['confidence_score'].mean():.3f}")
            print(f"平均ボラティリティ比率: {valid_data['volatility_ratio'].mean():.3f}")
            print(f"平均ノイズレベル: {valid_data['noise_level'].mean():.3f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Quantum Adaptive Flow Channel (QAFC) の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--process-noise', type=float, default=0.01, help='プロセスノイズ')
    parser.add_argument('--measurement-noise', type=float, default=0.1, help='観測ノイズ')
    parser.add_argument('--noise-window', type=int, default=20, help='ノイズ分析ウィンドウ')
    parser.add_argument('--prediction-lookback', type=int, default=10, help='予測ルックバック期間')
    parser.add_argument('--base-multiplier', type=float, default=2.0, help='基本チャネル幅倍率')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = QAFCChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        noise_window=args.noise_window,
        prediction_lookback=args.prediction_lookback,
        base_multiplier=args.base_multiplier,
        src_type=args.src_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output,
        show_volume=not args.no_volume
    )


if __name__ == "__main__":
    main()