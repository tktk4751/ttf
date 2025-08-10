#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from matplotlib.gridspec import GridSpec

# パスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データ取得のための依存関係
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError:
    print("データローダーが見つかりません。サンプルデータを使用します。")
    DataLoader = None

# インジケーター
from indicators.ultimate_choppiness_index import UltimateChoppinessIndex
from indicators.choppiness import ChoppinessIndex


class UltimateChoppinessChart:
    """
    Ultimate Choppiness Indexと従来のChoppiness Indexを比較表示するチャートクラス
    
    - ローソク足と出来高
    - Traditional Choppiness Index
    - Ultimate Choppiness Index (Fixed/Dynamic)
    - トレンド状態の視覚化
    - STR値と動的期間の表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.traditional_chop = None
        self.ultimate_chop_fixed = None
        self.ultimate_chop_dynamic = None
        self.fig = None
        self.axes = None
    
    def generate_sample_data(self, n_points: int = 1000) -> pd.DataFrame:
        """サンプル市場データを生成"""
        dates = pd.date_range(start='2023-01-01', periods=n_points, freq='4h')
        
        # トレンドとレンジを含む価格データを生成
        np.random.seed(42)
        base_price = 100.0
        prices = []
        
        for i in range(n_points):
            if i < n_points * 0.3:  # 最初の30%はトレンド上昇
                trend = 0.1 * i
                noise = np.random.normal(0, 0.5)
            elif i < n_points * 0.5:  # 次の20%はレンジ
                trend = 0.1 * n_points * 0.3
                noise = np.random.normal(0, 1.0) * 2
            elif i < n_points * 0.8:  # 次の30%はトレンド下降
                trend = 0.1 * n_points * 0.3 - 0.05 * (i - n_points * 0.5)
                noise = np.random.normal(0, 0.5)
            else:  # 最後の20%は再びレンジ
                trend = 0.1 * n_points * 0.3 - 0.05 * n_points * 0.3
                noise = np.random.normal(0, 1.0) * 2
            
            price = base_price + trend + noise
            prices.append(max(price, base_price * 0.5))
        
        prices = np.array(prices)
        
        # OHLCデータを生成
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['high'] = df['close'] + np.abs(np.random.normal(0, 0.5, n_points))
        df['low'] = df['close'] - np.abs(np.random.normal(0, 0.5, n_points))
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']
        df['volume'] = np.random.uniform(1000, 10000, n_points)
        
        return df
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        if DataLoader is None:
            print("データローダーが利用できません。サンプルデータを生成します。")
            self.data = self.generate_sample_data(n_points=1500)
            return self.data
        
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

    def calculate_indicators(self,
                           # Traditional Choppiness
                           traditional_period: int = 14,
                           # Ultimate Choppiness Fixed
                           ultimate_period: float = 55.0,
                           smooth_period: int = 3,
                           # Ultimate Choppiness Dynamic
                           cycle_detector_type: str = 'absolute_ultimate',
                           cycle_detector_cycle_part: float = 0.5,
                           cycle_detector_max_cycle: int = 55,
                           cycle_detector_min_cycle: int = 5,
                           cycle_period_multiplier: float = 1.0
                          ) -> None:
        """
        Traditional ChoppinessとUltimate Choppinessを計算する
        
        Args:
            traditional_period: Traditional Choppinessの期間
            ultimate_period: Ultimate Choppinessの基本期間
            smooth_period: Ultimate Choppinessの平滑化期間
            cycle_detector_type: サイクル検出器タイプ
            cycle_detector_cycle_part: サイクル部分倍率
            cycle_detector_max_cycle: 最大サイクル期間
            cycle_detector_min_cycle: 最小サイクル期間
            cycle_period_multiplier: サイクル期間の乗数
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nインジケーターを計算中...")
        
        # Traditional Choppiness Index
        print("Traditional Choppiness Indexを計算中...")
        self.traditional_chop = ChoppinessIndex(period=traditional_period)
        self.traditional_values = self.traditional_chop.calculate(self.data)
        
        # Ultimate Choppiness Index (Fixed)
        print("Ultimate Choppiness Index (Fixed)を計算中...")
        self.ultimate_chop_fixed = UltimateChoppinessIndex(
            period=ultimate_period,
            period_mode='fixed',
            smooth_period=smooth_period,
            trend_threshold=0.5,  # 新しい閾値
            range_threshold=0.5
        )
        self.ultimate_result_fixed = self.ultimate_chop_fixed.calculate(self.data)
        
        # Ultimate Choppiness Index (Dynamic)
        print("Ultimate Choppiness Index (Dynamic)を計算中...")
        self.ultimate_chop_dynamic = UltimateChoppinessIndex(
            period=55,
            period_mode='Fixed',
            smooth_period=smooth_period,
            trend_threshold=0.5,  # 新しい閾値
            range_threshold=0.5,
            cycle_detector_type=cycle_detector_type,
            cycle_detector_cycle_part=cycle_detector_cycle_part,
            cycle_detector_max_cycle=cycle_detector_max_cycle,
            cycle_detector_min_cycle=cycle_detector_min_cycle,
            cycle_period_multiplier=cycle_period_multiplier
        )
        self.ultimate_result_dynamic = self.ultimate_chop_dynamic.calculate(self.data)
        
        print("インジケーター計算完了")
        
        # トレンド判定率の分析
        self._analyze_trend_detection()
    
    def _analyze_trend_detection(self) -> None:
        """トレンド検出分析"""
        print("\n=== トレンド検出分析 ===")
        
        # Traditional Choppiness (38.2以下がトレンド)
        trend_traditional = self.traditional_values <= 38.2
        trend_ratio_traditional = np.sum(trend_traditional) / len(self.traditional_values) * 100
        
        # Ultimate Fixed (trend_state == 1がトレンド)
        trend_ratio_fixed = np.sum(self.ultimate_result_fixed.trend_state == 1) / len(self.ultimate_result_fixed.trend_state) * 100
        
        # Ultimate Dynamic
        trend_ratio_dynamic = np.sum(self.ultimate_result_dynamic.trend_state == 1) / len(self.ultimate_result_dynamic.trend_state) * 100
        
        print(f"トレンド判定率:")
        print(f"Traditional Choppiness: {trend_ratio_traditional:.2f}%")
        print(f"Ultimate Choppiness (Fixed): {trend_ratio_fixed:.2f}%")
        print(f"Ultimate Choppiness (Dynamic): {trend_ratio_dynamic:.2f}%")
        
        # 平均値分析
        print(f"\n平均チョピネス値:")
        print(f"Traditional: {np.nanmean(self.traditional_values):.2f}")
        print(f"Ultimate (Fixed): {np.nanmean(self.ultimate_result_fixed.values):.2f}")
        print(f"Ultimate (Dynamic): {np.nanmean(self.ultimate_result_dynamic.values):.2f}")
            
    def plot(self, 
            title: str = "Ultimate Choppiness Index Comparison", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとChoppiness Indexを描画する
        
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
            raise ValueError("データが読み込まれていません。")
            
        if self.traditional_values is None:
            raise ValueError("インジケーターが計算されていません。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インデックスの調整
        start_idx = len(self.data) - len(df)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(index=self.data.index)
        full_df['traditional'] = self.traditional_values
        full_df['ultimate_fixed'] = self.ultimate_result_fixed.values
        full_df['ultimate_dynamic'] = self.ultimate_result_dynamic.values
        full_df['trend_fixed'] = self.ultimate_result_fixed.trend_state
        full_df['trend_dynamic'] = self.ultimate_result_dynamic.trend_state
        full_df['dynamic_periods'] = self.ultimate_result_dynamic.dynamic_periods
        full_df['str_values'] = self.ultimate_result_dynamic.str_values
        
        # 絞り込み後のデータに結合
        df = df.join(full_df)
        
        # トレンド期間用のバックグラウンドカラー設定
        trend_mask_dynamic = df['trend_dynamic'] == 1
        
        # カスタムプロット作成
        fig = plt.figure(figsize=figsize)
        
        # GridSpec設定（出来高ありの場合）
        if show_volume:
            gs = GridSpec(6, 1, height_ratios=[3, 1, 1.5, 1.5, 1.5, 1], hspace=0.3)
        else:
            gs = GridSpec(5, 1, height_ratios=[3, 1.5, 1.5, 1.5, 1], hspace=0.3)
        
        # 価格チャート
        ax1 = fig.add_subplot(gs[0])
        
        # ローソク足データの準備
        ohlc = df[['open', 'high', 'low', 'close']].values
        dates = df.index
        
        # ローソク足を描画
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        for i in range(len(df)):
            date = i
            open_price = ohlc[i, 0]
            high_price = ohlc[i, 1]
            low_price = ohlc[i, 2]
            close_price = ohlc[i, 3]
            
            # 陽線・陰線の判定
            if close_price >= open_price:
                color = 'green'
                body_height = close_price - open_price
                body_bottom = open_price
            else:
                color = 'red'
                body_height = open_price - close_price
                body_bottom = close_price
            
            # ローソク足の実体
            rect = Rectangle((date - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor=color, alpha=0.8)
            ax1.add_patch(rect)
            
            # ヒゲ
            ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
        
        # トレンド期間のハイライト
        for i in range(1, len(trend_mask_dynamic)):
            if trend_mask_dynamic.iloc[i] and not trend_mask_dynamic.iloc[i-1]:
                start_idx = i
            elif not trend_mask_dynamic.iloc[i] and trend_mask_dynamic.iloc[i-1]:
                ax1.axvspan(start_idx, i-1, alpha=0.2, color='green')
        
        ax1.set_title(f'{title} - Price Chart with Trend Periods', fontsize=14)
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, len(df))
        ax1.set_xticks([])
        
        # 出来高
        ax_idx = 1
        if show_volume:
            ax2 = fig.add_subplot(gs[ax_idx])
            ax2.bar(range(len(df)), df['volume'], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-1, len(df))
            ax2.set_xticks([])
            ax_idx += 1
        
        # Traditional Choppiness
        ax3 = fig.add_subplot(gs[ax_idx])
        ax3.plot(range(len(df)), df['traditional'], 'b-', linewidth=1, label='Traditional')
        ax3.axhline(y=38.2, color='g', linestyle='--', alpha=0.5, label='Trend (38.2)')
        ax3.axhline(y=61.8, color='r', linestyle='--', alpha=0.5, label='Range (61.8)')
        ax3.fill_between(range(len(df)), 0, 38.2, alpha=0.1, color='green')
        ax3.fill_between(range(len(df)), 61.8, 100, alpha=0.1, color='red')
        ax3.set_ylabel('Choppiness')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Traditional Choppiness Index')
        ax3.set_xlim(-1, len(df))
        ax3.set_xticks([])
        ax_idx += 1
        
        # Ultimate Choppiness (Fixed)
        ax4 = fig.add_subplot(gs[ax_idx])
        ax4.plot(range(len(df)), df['ultimate_fixed'], 'purple', linewidth=1, label='Ultimate (Fixed)')
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Threshold (50)')
        ax4.fill_between(range(len(df)), 0, 50, alpha=0.1, color='green')
        ax4.fill_between(range(len(df)), 50, 100, alpha=0.1, color='red')
        
        # トレンド状態マーカー
        trend_points = df['trend_fixed'] == 1
        ax4.scatter(np.where(trend_points)[0], df['ultimate_fixed'][trend_points], 
                   color='green', s=10, alpha=0.5, label='Trend')
        
        ax4.set_ylabel('Choppiness')
        ax4.set_ylim(0, 100)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Ultimate Choppiness Index (Fixed Period)')
        ax4.set_xlim(-1, len(df))
        ax4.set_xticks([])
        ax_idx += 1
        
        # Ultimate Choppiness (Dynamic)
        ax5 = fig.add_subplot(gs[ax_idx])
        ax5.plot(range(len(df)), df['ultimate_dynamic'], 'orange', linewidth=1, label='Ultimate (Dynamic)')
        ax5.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Threshold (50)')
        ax5.fill_between(range(len(df)), 0, 50, alpha=0.1, color='green')
        ax5.fill_between(range(len(df)), 50, 100, alpha=0.1, color='red')
        
        # トレンド状態マーカー
        trend_points_dynamic = df['trend_dynamic'] == 1
        ax5.scatter(np.where(trend_points_dynamic)[0], df['ultimate_dynamic'][trend_points_dynamic], 
                   color='green', s=10, alpha=0.5, label='Trend')
        
        ax5.set_ylabel('Choppiness')
        ax5.set_ylim(0, 100)
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Ultimate Choppiness Index (Dynamic Period)')
        ax5.set_xlim(-1, len(df))
        ax5.set_xticks([])
        ax_idx += 1
        
        # 動的期間
        ax6 = fig.add_subplot(gs[ax_idx])
        ax6.plot(range(len(df)), df['dynamic_periods'], 'gray', linewidth=1)
        ax6.axhline(y=14, color='blue', linestyle='--', alpha=0.5, label='Base Period (14)')
        ax6.set_ylabel('Period')
        ax6.set_xlabel('Time')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_title('Dynamic Period (Ultimate Dynamic)')
        ax6.set_xlim(-1, len(df))
        
        # X軸ラベルの設定（最下段のみ）
        n_labels = 10
        indices = np.linspace(0, len(df)-1, n_labels, dtype=int)
        labels = [df.index[i].strftime('%Y-%m-%d') for i in indices]
        ax6.set_xticks(indices)
        ax6.set_xticklabels(labels, rotation=45)
        
        # 統計情報の表示
        print(f"\n=== チャート期間の統計 ===")
        print(f"期間: {df.index[0].strftime('%Y-%m-%d')} から {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"データ点数: {len(df)}")
        
        # 相関分析
        mask = ~np.isnan(df['traditional']) & ~np.isnan(df['ultimate_fixed']) & ~np.isnan(df['ultimate_dynamic'])
        corr_trad_fixed = np.corrcoef(df['traditional'][mask], df['ultimate_fixed'][mask])[0, 1]
        corr_trad_dynamic = np.corrcoef(df['traditional'][mask], df['ultimate_dynamic'][mask])[0, 1]
        corr_fixed_dynamic = np.corrcoef(df['ultimate_fixed'][mask], df['ultimate_dynamic'][mask])[0, 1]
        
        print(f"\n相関係数:")
        print(f"Traditional vs Ultimate (Fixed): {corr_trad_fixed:.4f}")
        print(f"Traditional vs Ultimate (Dynamic): {corr_trad_dynamic:.4f}")
        print(f"Ultimate (Fixed) vs Ultimate (Dynamic): {corr_fixed_dynamic:.4f}")
        
        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        else:
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Ultimate Choppiness Indexの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--traditional-period', type=int, default=14, help='Traditional Choppinessの期間')
    parser.add_argument('--ultimate-period', type=float, default=14.0, help='Ultimate Choppinessの期間')
    parser.add_argument('--smooth-period', type=int, default=3, help='平滑化期間')
    parser.add_argument('--detector-type', type=str, default='absolute_ultimate', help='サイクル検出器タイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateChoppinessChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        traditional_period=args.traditional_period,
        ultimate_period=args.ultimate_period,
        smooth_period=args.smooth_period,
        cycle_detector_type=args.detector_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()