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

# 統合サイクル検出器
try:
    from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError as e:
    print(f"Warning: Could not import EhlersUnifiedDC: {e}")
    print("Please check the import dependencies.")
    exit(1)


class EhlersUnifiedDCChart:
    """
    エーラーズ統合サイクル検出器の全検出器をテストするチャートクラス
    
    - ローソク足と出来高
    - 各サイクル検出器の結果を複数パネルで表示
    - 検出器の比較分析
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.detectors = {}
        self.results = {}
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

    def get_all_detectors(self) -> List[str]:
        """
        利用可能な全サイクル検出器を取得する
        
        Returns:
            検出器名のリスト
        """
        # EhlersUnifiedDCから利用可能な検出器を取得
        available_detectors = EhlersUnifiedDC.get_available_detectors()
        return list(available_detectors.keys())

    def calculate_all_detectors(self, 
                               src_type: str = 'hlc3',
                               max_cycle: int = 50,
                               min_cycle: int = 6,
                               max_output: int = 34,
                               min_output: int = 1,
                               cycle_part: float = 0.5,
                               use_kalman_filter: bool = False,
                               period_range: Tuple[int, int] = (5, 120)) -> None:
        """
        すべてのサイクル検出器を計算する
        
        Args:
            src_type: ソースタイプ
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            cycle_part: サイクル部分
            use_kalman_filter: カルマンフィルター使用有無
            period_range: 周期範囲
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n全サイクル検出器を計算中...")
        
        # 利用可能な検出器を取得
        detector_types = self.get_all_detectors()
        print(f"利用可能な検出器数: {len(detector_types)}")
        
        # 各検出器の説明を取得
        detector_descriptions = EhlersUnifiedDC.get_available_detectors()
        
        # 各検出器を計算
        for detector_type in detector_types:
            try:
                print(f"検出器 '{detector_type}' を計算中...")
                print(f"  説明: {detector_descriptions.get(detector_type, 'N/A')}")
                
                # 検出器を初期化
                detector = EhlersUnifiedDC(
                    detector_type=detector_type,
                    src_type=src_type,
                    max_cycle=max_cycle,
                    min_cycle=min_cycle,
                    max_output=max_output,
                    min_output=min_output,
                    cycle_part=cycle_part,
                    use_kalman_filter=use_kalman_filter,
                    period_range=period_range
                )
                
                # 計算実行
                result = detector.calculate(self.data)
                
                # 結果を保存
                self.detectors[detector_type] = detector
                self.results[detector_type] = result
                
                # 統計情報を表示
                valid_count = np.sum(~np.isnan(result))
                mean_value = np.nanmean(result)
                std_value = np.nanstd(result)
                print(f"  結果: 有効値数={valid_count}, 平均={mean_value:.2f}, 標準偏差={std_value:.2f}")
                
            except Exception as e:
                print(f"  エラー: {str(e)}")
                # エラーの場合はNaNで埋める
                self.results[detector_type] = np.full(len(self.data), np.nan)
                continue
        
        print(f"\n計算完了: {len(self.results)}個の検出器")
            
    def plot(self, 
            title: str = "エーラーズ統合サイクル検出器 - 全検出器比較", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 20),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            detectors_per_panel: int = 4) -> None:
        """
        ローソク足チャートと全サイクル検出器の結果を描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            detectors_per_panel: 1パネルあたりの検出器数
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if not self.results:
            raise ValueError("検出器が計算されていません。calculate_all_detectors()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 検出器の結果をDataFrameに追加
        print("チャートデータを準備中...")
        for detector_type, result in self.results.items():
            # インデックスを合わせて結果を追加
            full_result_df = pd.DataFrame(
                index=self.data.index,
                data={f'cycle_{detector_type}': result}
            )
            df = df.join(full_result_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # カラーパレットの定義
        colors = [
            'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
            'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'teal', 'maroon',
            'fuchsia', 'silver', 'darkgreen', 'darkred', 'darkblue', 'darkorange'
        ]
        
        # 検出器リストの準備
        detector_types = list(self.results.keys())
        
        # 追加プロットの準備
        addplots = []
        
        # パネル番号の管理
        panel_num = 1 if show_volume else 0
        
        # 検出器をパネルごとにグループ化
        detector_groups = [
            detector_types[i:i + detectors_per_panel] 
            for i in range(0, len(detector_types), detectors_per_panel)
        ]
        
        # 各パネルにプロットを追加
        for group_idx, detector_group in enumerate(detector_groups):
            for idx, detector_type in enumerate(detector_group):
                color = colors[idx % len(colors)]
                column_name = f'cycle_{detector_type}'
                
                if column_name in df.columns:
                    addplot = mpf.make_addplot(
                        df[column_name], 
                        panel=panel_num,
                        color=color,
                        width=1.2,
                        label=detector_type,
                        secondary_y=False
                    )
                    addplots.append(addplot)
            
            panel_num += 1
        
        # パネル比率の計算
        num_panels = len(detector_groups) + (1 if show_volume else 0)
        if show_volume:
            panel_ratios = [4, 1] + [2] * len(detector_groups)  # メイン:出来高:検出器パネル...
        else:
            panel_ratios = [4] + [2] * len(detector_groups)  # メイン:検出器パネル...
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            volume=show_volume,
            addplot=addplots,
            panel_ratios=panel_ratios,
            returnfig=True
        )
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 各パネルにタイトルと凡例を追加
        panel_idx = 1 if show_volume else 0
        for group_idx, detector_group in enumerate(detector_groups):
            if panel_idx < len(axes):
                # パネルタイトルの設定
                group_names = ', '.join(detector_group[:3])  # 最初の3つの名前のみ表示
                if len(detector_group) > 3:
                    group_names += f" (他{len(detector_group) - 3}個)"
                axes[panel_idx].set_title(f"パネル {group_idx + 1}: {group_names}", fontsize=10)
                
                # Y軸ラベル
                axes[panel_idx].set_ylabel('サイクル値', fontsize=9)
                
                # 参照線を追加
                y_min, y_max = axes[panel_idx].get_ylim()
                axes[panel_idx].axhline(y=np.mean([y_min, y_max]), color='black', linestyle='--', alpha=0.3)
                
            panel_idx += 1
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n=== サイクル検出器統計 ===")
        for detector_type, result in self.results.items():
            valid_count = np.sum(~np.isnan(result))
            if valid_count > 0:
                mean_val = np.nanmean(result)
                std_val = np.nanstd(result)
                min_val = np.nanmin(result)
                max_val = np.nanmax(result)
                print(f"{detector_type:20}: 平均={mean_val:6.2f}, 標準偏差={std_val:6.2f}, 範囲=[{min_val:6.2f}, {max_val:6.2f}]")
            else:
                print(f"{detector_type:20}: データなし")
        
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
    parser = argparse.ArgumentParser(description='エーラーズ統合サイクル検出器の全検出器テスト')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--detectors-per-panel', type=int, default=4, help='1パネルあたりの検出器数')
    parser.add_argument('--max-cycle', type=int, default=50, help='最大サイクル期間')
    parser.add_argument('--min-cycle', type=int, default=6, help='最小サイクル期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = EhlersUnifiedDCChart()
    chart.load_data_from_config(args.config)
    chart.calculate_all_detectors(
        src_type=args.src_type,
        max_cycle=args.max_cycle,
        min_cycle=args.min_cycle
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output,
        detectors_per_panel=args.detectors_per_panel
    )


if __name__ == "__main__":
    main()