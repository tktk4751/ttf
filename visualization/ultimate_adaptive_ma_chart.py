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
from indicators.ultimate_adaptive_ma import UltimateAdaptiveMA


class UltimateAdaptiveMAChart:
    """
    Ultimate Adaptive MAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Ultimate Adaptive MA（主線）
    - ベース移動平均（比較用）
    - FRAMA・MAMA値（サブパネル）
    - 適応ファクター・トレンド強度・応答性（オシレータパネル）
    - マーケットレジーム表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_adaptive_ma = None
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
                            # ベースMA設定
                            base_period: int = 21,
                            src_type: str = 'hlc3',
                            # FRAMA設定
                            frama_period: int = 16,
                            frama_fc: int = 1,
                            frama_sc: int = 198,
                            # MAMA設定
                            mama_fast_limit: float = 0.5,
                            mama_slow_limit: float = 0.05,
                            # Ultimate Smoother設定
                            smoother_period: float = 5.0,
                            # 適応パラメータ
                            adaptation_strength: float = 0.8,
                            min_alpha: float = 0.05,
                            max_alpha: float = 0.8,
                            volatility_period: int = 14,
                            # 動的期間パラメータ
                            use_dynamic_periods: bool = True,
                            cycle_detector_type: str = 'hody_e') -> None:
        """
        Ultimate Adaptive MAを計算する
        
        Args:
            base_period: ベース移動平均期間
            src_type: 価格ソース
            frama_period: FRAMA期間
            frama_fc: FRAMA高速定数
            frama_sc: FRAMA低速定数
            mama_fast_limit: MAMA高速リミット
            mama_slow_limit: MAMA低速リミット
            smoother_period: Ultimate Smoother期間
            adaptation_strength: 適応強度
            min_alpha: 最小アルファ値
            max_alpha: 最大アルファ値
            volatility_period: ボラティリティ計算期間
            use_dynamic_periods: 動的期間使用
            cycle_detector_type: サイクル検出器タイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nUltimate Adaptive MAを計算中...")
        
        # Ultimate Adaptive MAを計算
        self.ultimate_adaptive_ma = UltimateAdaptiveMA(
            base_period=base_period,
            src_type=src_type,
            frama_period=frama_period,
            frama_fc=frama_fc,
            frama_sc=frama_sc,
            mama_fast_limit=mama_fast_limit,
            mama_slow_limit=mama_slow_limit,
            smoother_period=smoother_period,
            adaptation_strength=adaptation_strength,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            volatility_period=volatility_period,
            use_dynamic_periods=use_dynamic_periods,
            cycle_detector_type=cycle_detector_type
        )
        
        # Ultimate Adaptive MAの計算
        print("計算を実行します...")
        result = self.ultimate_adaptive_ma.calculate(self.data)
        
        # 結果の取得と検証
        ultimate_ma = self.ultimate_adaptive_ma.get_values()
        base_ma = self.ultimate_adaptive_ma.get_base_ma()
        adaptive_factor = self.ultimate_adaptive_ma.get_adaptive_factor()
        trend_strength = self.ultimate_adaptive_ma.get_trend_strength()
        
        print(f"Ultimate Adaptive MA計算完了")
        print(f"データ長: {len(ultimate_ma) if ultimate_ma is not None else 0}")
        print(f"NaN値: {np.isnan(ultimate_ma).sum() if ultimate_ma is not None else 'N/A'}")
        
        # 統計情報
        if ultimate_ma is not None and len(ultimate_ma) > 0:
            valid_data = ultimate_ma[~np.isnan(ultimate_ma)]
            if len(valid_data) > 0:
                print(f"Ultimate MA範囲: {valid_data.min():.2f} - {valid_data.max():.2f}")
                
        if adaptive_factor is not None and len(adaptive_factor) > 0:
            valid_factor = adaptive_factor[~np.isnan(adaptive_factor)]
            if len(valid_factor) > 0:
                print(f"適応ファクター範囲: {valid_factor.min():.3f} - {valid_factor.max():.3f}")
                print(f"平均適応ファクター: {valid_factor.mean():.3f}")
        
        print("Ultimate Adaptive MA計算完了")
            
    def plot(self, 
            title: str = "Ultimate Adaptive MA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUltimate Adaptive MAを描画する
        
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
            
        if self.ultimate_adaptive_ma is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Ultimate Adaptive MAの値を取得
        print("インジケーターデータを取得中...")
        ultimate_ma = self.ultimate_adaptive_ma.get_values()
        base_ma = self.ultimate_adaptive_ma.get_base_ma()
        adaptive_factor = self.ultimate_adaptive_ma.get_adaptive_factor()
        trend_strength = self.ultimate_adaptive_ma.get_trend_strength()
        market_regime = self.ultimate_adaptive_ma.get_market_regime()
        responsiveness = self.ultimate_adaptive_ma.get_responsiveness()
        
        # FRAMAとMAMAデータの取得
        frama_data = self.ultimate_adaptive_ma.get_frama_data()
        mama_data = self.ultimate_adaptive_ma.get_mama_data()
        
        frama_values = frama_data[0] if frama_data else np.full(len(self.data), np.nan)
        frama_alpha = frama_data[1] if frama_data else np.full(len(self.data), np.nan)
        fractal_dimension = frama_data[2] if frama_data else np.full(len(self.data), np.nan)
        
        mama_values = mama_data[0] if mama_data else np.full(len(self.data), np.nan)
        mama_alpha = mama_data[1] if mama_data else np.full(len(self.data), np.nan)
        cycle_period = mama_data[2] if mama_data else np.full(len(self.data), np.nan)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'ultimate_ma': ultimate_ma if ultimate_ma is not None else np.full(len(self.data), np.nan),
                'base_ma': base_ma if base_ma is not None else np.full(len(self.data), np.nan),
                'frama': frama_values,
                'mama': mama_values,
                'adaptive_factor': adaptive_factor if adaptive_factor is not None else np.full(len(self.data), np.nan),
                'trend_strength': trend_strength if trend_strength is not None else np.full(len(self.data), np.nan),
                'market_regime': market_regime if market_regime is not None else np.full(len(self.data), np.nan),
                'responsiveness': responsiveness if responsiveness is not None else np.full(len(self.data), np.nan),
                'frama_alpha': frama_alpha,
                'mama_alpha': mama_alpha,
                'fractal_dimension': fractal_dimension,
                'cycle_period': cycle_period
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"Ultimate MA NaN数: {df['ultimate_ma'].isna().sum()}")
        
        # マーケットレジームに基づくUltimate MAの色分け
        df['ultimate_ma_trend'] = np.where(df['market_regime'] == 1, df['ultimate_ma'], np.nan)
        df['ultimate_ma_range'] = np.where(df['market_regime'] == 0, df['ultimate_ma'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # Ultimate Adaptive MA（レジーム別色分け）
        main_plots.append(mpf.make_addplot(df['ultimate_ma_trend'], color='green', width=3, label='Ultimate MA (Trend)'))
        main_plots.append(mpf.make_addplot(df['ultimate_ma_range'], color='orange', width=3, label='Ultimate MA (Range)'))
        
        # ベース移動平均（比較用）
        main_plots.append(mpf.make_addplot(df['base_ma'], color='blue', width=1, alpha=0.7, label='Base MA'))
        
        # FRAMA・MAMA（薄い色で表示）
        main_plots.append(mpf.make_addplot(df['frama'], color='purple', width=1, alpha=0.5, label='FRAMA'))
        main_plots.append(mpf.make_addplot(df['mama'], color='cyan', width=1, alpha=0.5, label='MAMA'))
        
        # 2. オシレータープロット
        # 適応ファクター・トレンド強度パネル
        adaptation_panel = mpf.make_addplot(df['adaptive_factor'], panel=1, color='red', width=1.5, 
                                          ylabel='Adaptation Factor', secondary_y=False, label='Adaptive Factor')
        trend_panel = mpf.make_addplot(df['trend_strength'], panel=1, color='green', width=1.5, 
                                     secondary_y=True, label='Trend Strength')
        
        # 応答性・マーケットレジームパネル
        responsiveness_panel = mpf.make_addplot(df['responsiveness'], panel=2, color='blue', width=1.5, 
                                               ylabel='Responsiveness', secondary_y=False, label='Responsiveness')
        regime_panel = mpf.make_addplot(df['market_regime'], panel=2, color='orange', width=2, 
                                       secondary_y=True, label='Market Regime', type='line')
        
        # フラクタル次元・サイクル期間パネル
        fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=3, color='purple', width=1.5, 
                                        ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
        cycle_panel = mpf.make_addplot(df['cycle_period'], panel=3, color='cyan', width=1.5, 
                                      secondary_y=True, label='Cycle Period')
        
        # Alpha値パネル
        frama_alpha_panel = mpf.make_addplot(df['frama_alpha'], panel=4, color='purple', width=1.5, 
                                            ylabel='Alpha Values', secondary_y=False, label='FRAMA Alpha')
        mama_alpha_panel = mpf.make_addplot(df['mama_alpha'], panel=4, color='cyan', width=1.5, 
                                           secondary_y=False, label='MAMA Alpha')
        
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
            kwargs['panel_ratios'] = (5, 1, 1.5, 1.5, 1.5, 1)  # メイン:出来高:適応:応答性:フラクタル:アルファ
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            adaptation_panel = mpf.make_addplot(df['adaptive_factor'], panel=2, color='red', width=1.5, 
                                              ylabel='Adaptation Factor', secondary_y=False, label='Adaptive Factor')
            trend_panel = mpf.make_addplot(df['trend_strength'], panel=2, color='green', width=1.5, 
                                         secondary_y=True, label='Trend Strength')
            responsiveness_panel = mpf.make_addplot(df['responsiveness'], panel=3, color='blue', width=1.5, 
                                                   ylabel='Responsiveness', secondary_y=False, label='Responsiveness')
            regime_panel = mpf.make_addplot(df['market_regime'], panel=3, color='orange', width=2, 
                                           secondary_y=True, label='Market Regime', type='line')
            fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=4, color='purple', width=1.5, 
                                            ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
            cycle_panel = mpf.make_addplot(df['cycle_period'], panel=4, color='cyan', width=1.5, 
                                          secondary_y=True, label='Cycle Period')
            frama_alpha_panel = mpf.make_addplot(df['frama_alpha'], panel=5, color='purple', width=1.5, 
                                                ylabel='Alpha Values', secondary_y=False, label='FRAMA Alpha')
            mama_alpha_panel = mpf.make_addplot(df['mama_alpha'], panel=5, color='cyan', width=1.5, 
                                               secondary_y=False, label='MAMA Alpha')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1.5, 1.5, 1.5, 1)  # メイン:適応:応答性:フラクタル:アルファ
        
        # すべてのプロットを結合
        all_plots = main_plots + [
            adaptation_panel, trend_panel, responsiveness_panel, regime_panel,
            fractal_panel, cycle_panel, frama_alpha_panel, mama_alpha_panel
        ]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Ultimate MA (Trend)', 'Ultimate MA (Range)', 'Base MA', 'FRAMA', 'MAMA'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # 適応ファクター・トレンド強度パネル
        axes[panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 応答性・マーケットレジームパネル
        axes[panel_offset + 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset + 1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset + 1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # フラクタル次元パネル
        axes[panel_offset + 2].axhline(y=1.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset + 2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        axes[panel_offset + 2].axhline(y=2.0, color='red', linestyle='--', alpha=0.5)
        
        # Alpha値パネル
        axes[panel_offset + 3].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset + 3].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset + 3].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== Ultimate Adaptive MA統計 ===")
        valid_data = df.dropna(subset=['ultimate_ma', 'adaptive_factor', 'trend_strength'])
        
        if len(valid_data) > 0:
            trend_points = len(valid_data[valid_data['market_regime'] == 1])
            range_points = len(valid_data[valid_data['market_regime'] == 0])
            total_points = len(valid_data)
            
            print(f"総データ点数: {total_points}")
            print(f"トレンド相場: {trend_points} ({trend_points/total_points*100:.1f}%)")
            print(f"レンジ相場: {range_points} ({range_points/total_points*100:.1f}%)")
            print(f"平均適応ファクター: {valid_data['adaptive_factor'].mean():.3f}")
            print(f"平均トレンド強度: {valid_data['trend_strength'].mean():.3f}")
            print(f"平均応答性: {valid_data['responsiveness'].mean():.3f}")
        
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
    parser = argparse.ArgumentParser(description='Ultimate Adaptive MAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--base-period', type=int, default=21, help='ベース移動平均期間')
    parser.add_argument('--adaptation-strength', type=float, default=0.8, help='適応強度')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateAdaptiveMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        base_period=args.base_period,
        adaptation_strength=args.adaptation_strength,
        src_type=args.src_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()