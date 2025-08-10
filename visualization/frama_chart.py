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

# FRAMAインジケーター
from indicators.smoother.frama import FRAMA


class FRAMAChart:
    """
    FRAMA (Fractal Adaptive Moving Average) を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - FRAMA値（適応的移動平均）
    - フラクタル次元（価格の複雑さ）
    - アルファ値（適応係数）
    - 異なる期間・ソースタイプの比較
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.frama_indicators = {}
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
                           frama_configs: List[Dict[str, Any]] = None
                           ) -> None:
        """
        FRAMAインジケーターを計算する
        
        Args:
            frama_configs: FRAMAの設定リスト
                例: [
                    {'period': 16, 'src_type': 'hl2', 'name': 'FRAMA_16_HL2'},
                    {'period': 32, 'src_type': 'hlc3', 'name': 'FRAMA_32_HLC3'}
                ]
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
        
        # デフォルトの設定
        if frama_configs is None:
            frama_configs = [
                {'period': 16, 'src_type': 'hl2', 'fc': 1, 'sc': 198, 'name': 'FRAMA_16_HL2'},
                {'period': 32, 'src_type': 'hlc3', 'fc': 1, 'sc': 198, 'name': 'FRAMA_32_HLC3'},
                {'period': 20, 'src_type': 'close', 'fc': 1, 'sc': 198, 'name': 'FRAMA_20_CLOSE'}
            ]
        
        print("\nFRAMAインジケーターを計算中...")
        
        for config in frama_configs:
            print(f"  計算中: {config['name']}")
            
            # FRAMAインジケーターを作成
            frama = FRAMA(
                period=config['period'],
                src_type=config['src_type'],
                fc=config.get('fc', 1),
                sc=config.get('sc', 198)
            )
            
            # 計算実行
            result = frama.calculate(self.data)
            
            # 結果を保存
            self.frama_indicators[config['name']] = {
                'indicator': frama,
                'result': result,
                'config': config
            }
            
            # 統計情報を表示
            valid_values = result.values[~np.isnan(result.values)]
            valid_dim = result.fractal_dimension[~np.isnan(result.fractal_dimension)]
            valid_alpha = result.alpha[~np.isnan(result.alpha)]
            
            print(f"    有効値数: {len(valid_values)}")
            print(f"    FRAMA範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
            print(f"    フラクタル次元平均: {np.mean(valid_dim):.4f}")
            print(f"    アルファ平均: {np.mean(valid_alpha):.4f}")
        
        print("FRAMAインジケーター計算完了")
            
    def plot(self, 
            title: str = "FRAMA (Fractal Adaptive Moving Average)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            show_fractal_dimension: bool = True,
            show_alpha: bool = True,
            show_plot: bool = True) -> None:
        """
        ローソク足チャートとFRAMAを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は保存しない）
            show_fractal_dimension: フラクタル次元を表示するか
            show_alpha: アルファ値を表示するか
            show_plot: matplotlibで表示するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if not self.frama_indicators:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # FRAMAの値を全データフレームに追加
        print("FRAMAデータを準備中...")
        for name, indicator_data in self.frama_indicators.items():
            result = indicator_data['result']
            config = indicator_data['config']
            
            # 全データの時系列データフレームを作成
            full_df = pd.DataFrame(
                index=self.data.index,
                data={
                    f'{name}_values': result.values,
                    f'{name}_fractal_dim': result.fractal_dimension,
                    f'{name}_alpha': result.alpha
                }
            )
            
            # 絞り込み後のデータに対してインジケーターデータを結合
            df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # FRAMA値のプロット設定
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        line_styles = ['-', '--', '-.', ':', '-']
        
        for i, (name, indicator_data) in enumerate(self.frama_indicators.items()):
            config = indicator_data['config']
            color = colors[i % len(colors)]
            style_line = line_styles[i % len(line_styles)]
            
            # FRAMA値をメインチャートに追加
            main_plots.append(
                mpf.make_addplot(
                    df[f'{name}_values'], 
                    color=color, 
                    width=2,
                    linestyle=style_line,
                    label=f"{name} (p={config['period']}, {config['src_type']})"
                )
            )
        
        # 追加パネルの設定
        additional_plots = []
        panel_count = 0
        
        # フラクタル次元のパネル
        if show_fractal_dimension:
            panel_count += 1
            for i, (name, indicator_data) in enumerate(self.frama_indicators.items()):
                color = colors[i % len(colors)]
                additional_plots.append(
                    mpf.make_addplot(
                        df[f'{name}_fractal_dim'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color=color, 
                        width=1.5,
                        ylabel='Fractal Dimension',
                        label=f"{name} Fractal Dim"
                    )
                )
        
        # アルファ値のパネル
        if show_alpha:
            panel_count += 1
            for i, (name, indicator_data) in enumerate(self.frama_indicators.items()):
                color = colors[i % len(colors)]
                additional_plots.append(
                    mpf.make_addplot(
                        df[f'{name}_alpha'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color=color, 
                        width=1.5,
                        ylabel='Alpha Values',
                        label=f"{name} Alpha"
                    )
                )
        
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
        
        # パネル比率の計算
        panel_ratios = [4]  # メインチャート
        
        if show_volume:
            panel_ratios.append(1)  # 出来高
        
        if show_fractal_dimension:
            panel_ratios.append(1.5)  # フラクタル次元
        
        if show_alpha:
            panel_ratios.append(1.5)  # アルファ値
        
        # 出来高と追加パネルの設定
        kwargs['volume'] = show_volume
        kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # すべてのプロットを結合
        all_plots = main_plots + additional_plots
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        frama_labels = []
        for name, indicator_data in self.frama_indicators.items():
            config = indicator_data['config']
            frama_labels.append(f"{name} (p={config['period']}, {config['src_type']})")
        
        axes[0].legend(frama_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 参照線の追加
        panel_idx = 1 if show_volume else 0
        
        # フラクタル次元の参照線
        if show_fractal_dimension:
            panel_idx += 1
            axes[panel_idx].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='D=1 (Fast)')
            axes[panel_idx].axhline(y=1.5, color='gray', linestyle=':', alpha=0.5, label='D=1.5 (Mid)')
            axes[panel_idx].axhline(y=2.0, color='black', linestyle='--', alpha=0.5, label='D=2 (Slow)')
            axes[panel_idx].set_ylim(0.8, 2.2)
        
        # アルファ値の参照線
        if show_alpha:
            panel_idx += 1
            axes[panel_idx].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min α=0.01')
            axes[panel_idx].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Mid α=0.5')
            axes[panel_idx].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max α=1.0')
            axes[panel_idx].set_ylim(-0.1, 1.1)
        
        # 統計情報の表示
        print(f"\n=== FRAMA統計情報 ===")
        for name, indicator_data in self.frama_indicators.items():
            result = indicator_data['result']
            config = indicator_data['config']
            
            # 表示期間に対応する統計を計算
            values_slice = df[f'{name}_values'].dropna()
            fractal_slice = df[f'{name}_fractal_dim'].dropna()
            alpha_slice = df[f'{name}_alpha'].dropna()
            
            print(f"\n{name}:")
            print(f"  設定: period={config['period']}, src_type={config['src_type']}")
            print(f"  FRAMA値: {values_slice.mean():.4f} ± {values_slice.std():.4f}")
            print(f"  フラクタル次元: {fractal_slice.mean():.4f} (範囲: {fractal_slice.min():.4f} - {fractal_slice.max():.4f})")
            print(f"  アルファ値: {alpha_slice.mean():.4f} (範囲: {alpha_slice.min():.4f} - {alpha_slice.max():.4f})")
            
            # 適応性の分析
            high_alpha_pct = (alpha_slice > 0.5).sum() / len(alpha_slice) * 100
            low_alpha_pct = (alpha_slice < 0.1).sum() / len(alpha_slice) * 100
            print(f"  高速モード(α>0.5): {high_alpha_pct:.1f}%")
            print(f"  低速モード(α<0.1): {low_alpha_pct:.1f}%")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        
        # matplotlib表示
        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='FRAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period1', type=int, default=16, help='1つ目のFRAMA期間')
    parser.add_argument('--period2', type=int, default=32, help='2つ目のFRAMA期間')
    parser.add_argument('--period3', type=int, default=20, help='3つ目のFRAMA期間')
    parser.add_argument('--src1', type=str, default='hl2', help='1つ目のソースタイプ')
    parser.add_argument('--src2', type=str, default='hlc3', help='2つ目のソースタイプ')
    parser.add_argument('--src3', type=str, default='close', help='3つ目のソースタイプ')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-fractal', action='store_true', help='フラクタル次元を非表示')
    parser.add_argument('--no-alpha', action='store_true', help='アルファ値を非表示')
    parser.add_argument('--no-show', action='store_true', help='matplotlibでの表示を無効化（保存のみ）')
    parser.add_argument('--fc', type=int, default=1, help='Fast Constant (デフォルト: 1)')
    parser.add_argument('--sc', type=int, default=198, help='Slow Constant (デフォルト: 198)')
    args = parser.parse_args()
    
    # FRAMA設定の作成
    frama_configs = [
        {'period': args.period1, 'src_type': args.src1, 'fc': args.fc, 'sc': args.sc, 'name': f'FRAMA_{args.period1}_{args.src1.upper()}'},
        {'period': args.period2, 'src_type': args.src2, 'fc': args.fc, 'sc': args.sc, 'name': f'FRAMA_{args.period2}_{args.src2.upper()}'},
        {'period': args.period3, 'src_type': args.src3, 'fc': args.fc, 'sc': args.sc, 'name': f'FRAMA_{args.period3}_{args.src3.upper()}'}
    ]
    
    # チャートを作成
    chart = FRAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(frama_configs=frama_configs)
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_fractal_dimension=not args.no_fractal,
        show_alpha=not args.no_alpha,
        show_plot=not args.no_show,
        savefig=args.output
    )


if __name__ == "__main__":
    main()