#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Unscented Kalman Filter (UKF) チャート可視化** 🎯

実際の相場データを使用した無香料カルマンフィルターの可視化：
- YAML設定ファイルからのデータ読み込み
- BinanceDataSourceを使用した実データ取得
- mplfinanceを使用した高品質チャート表示
- UKFフィルタリング結果のローソク足チャート重ね合わせ
- 複数パネルでの詳細分析表示
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # データ取得のための依存関係
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
    
    # インジケーター
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.price_source import PriceSource
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールが見つかりません。以下を確認してください：")
    print("- data/data_loader.py")
    print("- data/binance_data_source.py") 
    print("- indicators/unscented_kalman_filter.py")
    sys.exit(1)


class UKFChart:
    """
    無香料カルマンフィルターチャート可視化クラス
    
    - ローソク足と出来高表示
    - UKFフィルタリング結果の重ね合わせ
    - 速度・加速度推定の表示
    - 不確実性と信頼度の表示
    - 期間指定とパラメータ調整機能
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ukf = None
        self.ukf_result = None
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
        if not os.path.exists(config_path):
            print(f"設定ファイルが見つかりません: {config_path}")
            print("デフォルト設定ファイルを作成します...")
            self._create_default_config(config_path)
        
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print(f"設定ファイルを読み込みました: {config_path}")
        
        # データの準備
        binance_config = config.get('binance_data', {})
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
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            print("テストデータを生成します...")
            self.data = self._generate_test_data()
            
        return self.data
    
    def _create_default_config(self, config_path: str) -> None:
        """デフォルト設定ファイルを作成"""
        default_config = {
            'binance_data': {
                'data_dir': 'data/binance',
                'symbols': ['BTCUSDT'],
                'intervals': ['1h'],
                'start_date': '2024-01-01',
                'end_date': '2024-12-31',
                'data_source': 'binance'
            }
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"デフォルト設定ファイルを作成しました: {config_path}")
    
    def _generate_test_data(self, n_points: int = 1000) -> pd.DataFrame:
        """テスト用の価格データを生成"""
        print("テスト用データを生成中...")
        np.random.seed(42)
        
        # より現実的な価格データの生成
        prices = []
        price = 50000.0  # BTCの典型的な価格から開始
        volatility = 0.02
        
        # 日時インデックスの生成
        dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
        
        for i in range(n_points):
            # 市場開閉時間の影響をシミュレート
            hour = dates[i].hour
            if 6 <= hour <= 22:  # アクティブ時間
                vol_multiplier = 1.0
            else:  # 非アクティブ時間
                vol_multiplier = 0.5
            
            # トレンドとサイクル要素
            trend = 0.0001 * i
            cycle1 = 1000 * np.sin(2 * np.pi * i / 168)  # 週次サイクル
            cycle2 = 500 * np.sin(2 * np.pi * i / 24)    # 日次サイクル
            
            # ボラティリティクラスタリング
            if i % 100 == 0:  # 時々大きな変動
                shock = np.random.normal(0, volatility * 5)
            else:
                shock = 0
            
            # 価格変動
            noise = np.random.normal(0, volatility * vol_multiplier)
            price_change = trend + (cycle1 + cycle2) * 0.01 + noise + shock
            price = max(price + price_change, 1000.0)  # 最低価格制限
            prices.append(price)
        
        prices = np.array(prices)
        
        # OHLCV データの生成
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
        open_prices = np.roll(prices, 1)
        open_prices[0] = prices[0]
        volume = np.random.exponential(1000, n_points)  # 出来高
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=dates)
        
        print(f"テストデータ生成完了: {len(df)}行")
        return df

    def calculate_ukf(self,
                     src_type: str = 'close',
                     alpha: float = 0.001,
                     beta: float = 2.0,
                     kappa: float = 0.0,
                     process_noise_scale: float = 0.001,
                     adaptive_noise: bool = True) -> None:
        """
        無香料カルマンフィルターを計算する
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            alpha: UKFアルファパラメータ
            beta: UKFベータパラメータ
            kappa: UKFカッパパラメータ
            process_noise_scale: プロセスノイズスケール
            adaptive_noise: 適応的ノイズ推定を使用するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nUKFを計算中... (α={alpha}, β={beta}, κ={kappa})")
        
        # UKFインスタンス作成
        self.ukf = UnscentedKalmanFilter(
            src_type=src_type,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            process_noise_scale=process_noise_scale,
            adaptive_noise=adaptive_noise
        )
        
        # UKF計算実行
        self.ukf_result = self.ukf.calculate(self.data)
        
        print("UKF計算完了")
        print(f"  データ点数: {len(self.ukf_result.filtered_values)}")
        print(f"  平均不確実性: {np.nanmean(self.ukf_result.uncertainty):.6f}")
        print(f"  平均信頼度: {np.nanmean(self.ukf_result.confidence_scores):.4f}")
        print(f"  平均速度: {np.nanmean(np.abs(self.ukf_result.velocity_estimates)):.6f}")
        print(f"  平均加速度: {np.nanmean(np.abs(self.ukf_result.acceleration_estimates)):.6f}")
            
    def plot(self, 
            title: str = "無香料カルマンフィルター (UKF) チャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUKF結果を描画する
        
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
            
        if self.ukf_result is None:
            raise ValueError("UKFが計算されていません。calculate_ukf()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # UKF結果の時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'ukf_filtered': self.ukf_result.filtered_values,
                'ukf_velocity': self.ukf_result.velocity_estimates,
                'ukf_acceleration': self.ukf_result.acceleration_estimates,
                'ukf_uncertainty': self.ukf_result.uncertainty,
                'ukf_confidence': self.ukf_result.confidence_scores,
                'ukf_innovation': self.ukf_result.innovations,
                'ukf_kalman_gain': self.ukf_result.kalman_gains,
                'ukf_raw': self.ukf_result.raw_values
            }
        )
        
        # 絞り込み後のデータに対してUKF結果を結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"UKFデータ確認 - フィルター値NaN: {df['ukf_filtered'].isna().sum()}")
        
        # UKFフィルター結果とトレンド方向の分析
        # 速度に基づくトレンド判定
        df['trend_direction'] = np.where(df['ukf_velocity'] > 0, 1, 
                                       np.where(df['ukf_velocity'] < 0, -1, 0))
        
        # トレンド別のフィルター値表示用
        df['ukf_uptrend'] = np.where(df['trend_direction'] == 1, df['ukf_filtered'], np.nan)
        df['ukf_downtrend'] = np.where(df['trend_direction'] == -1, df['ukf_filtered'], np.nan)
        df['ukf_neutral'] = np.where(df['trend_direction'] == 0, df['ukf_filtered'], np.nan)
        
        # 不確実性バンドの計算
        df['ukf_upper_band'] = df['ukf_filtered'] + df['ukf_uncertainty']
        df['ukf_lower_band'] = df['ukf_filtered'] - df['ukf_uncertainty']
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # UKFフィルター結果のプロット（トレンド別色分け）
        main_plots.append(mpf.make_addplot(df['ukf_uptrend'], color='green', width=2.5, alpha=0.8, label='UKF Filter (Up)'))
        main_plots.append(mpf.make_addplot(df['ukf_downtrend'], color='red', width=2.5, alpha=0.8, label='UKF Filter (Down)'))
        main_plots.append(mpf.make_addplot(df['ukf_neutral'], color='blue', width=2.5, alpha=0.8, label='UKF Filter (Neutral)'))
        
        # 不確実性バンド
        main_plots.append(mpf.make_addplot(df['ukf_upper_band'], color='gray', width=1, alpha=0.4, linestyle='--', label='Uncertainty Band'))
        main_plots.append(mpf.make_addplot(df['ukf_lower_band'], color='gray', width=1, alpha=0.4, linestyle='--'))
        
        # 2. 副パネルのプロット設定
        # 速度パネル
        velocity_panel = mpf.make_addplot(df['ukf_velocity'], panel=1, color='blue', width=1.5, 
                                        ylabel='Velocity', secondary_y=False, label='Velocity')
        
        # 加速度パネル  
        acceleration_panel = mpf.make_addplot(df['ukf_acceleration'], panel=2, color='purple', width=1.5, 
                                            ylabel='Acceleration', secondary_y=False, label='Acceleration')
        
        # 不確実性と信頼度パネル
        uncertainty_panel = mpf.make_addplot(df['ukf_uncertainty'], panel=3, color='orange', width=1.5, 
                                           ylabel='Uncertainty', secondary_y=False, label='Uncertainty')
        confidence_panel = mpf.make_addplot(df['ukf_confidence'], panel=3, color='green', width=1.5, 
                                          secondary_y=True, label='Confidence')
        
        # カルマンゲインとイノベーションパネル
        gain_panel = mpf.make_addplot(df['ukf_kalman_gain'], panel=4, color='brown', width=1.5, 
                                    ylabel='Kalman Gain', secondary_y=False, label='Kalman Gain')
        innovation_panel = mpf.make_addplot(df['ukf_innovation'], panel=4, color='red', width=1.5, 
                                          secondary_y=True, label='Innovation')
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d %H:%M',
            xrotation=45,
            returnfig=True
        )
        
        # 出来高と追加パネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:速度:加速度:不確実性:ゲイン
            # 出来高を表示する場合は、副パネルの番号を+1する
            velocity_panel = mpf.make_addplot(df['ukf_velocity'], panel=2, color='blue', width=1.5, 
                                            ylabel='Velocity', label='Velocity')
            acceleration_panel = mpf.make_addplot(df['ukf_acceleration'], panel=3, color='purple', width=1.5, 
                                                ylabel='Acceleration', label='Acceleration')
            uncertainty_panel = mpf.make_addplot(df['ukf_uncertainty'], panel=4, color='orange', width=1.5, 
                                               ylabel='Uncertainty', label='Uncertainty')
            confidence_panel = mpf.make_addplot(df['ukf_confidence'], panel=4, color='green', width=1.5, 
                                              secondary_y=True, label='Confidence')
            gain_panel = mpf.make_addplot(df['ukf_kalman_gain'], panel=5, color='brown', width=1.5, 
                                        ylabel='Kalman Gain', label='Kalman Gain')
            innovation_panel = mpf.make_addplot(df['ukf_innovation'], panel=5, color='red', width=1.5, 
                                              secondary_y=True, label='Innovation')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:速度:加速度:不確実性:ゲイン
        
        # すべてのプロットを結合
        all_plots = main_plots + [velocity_panel, acceleration_panel, uncertainty_panel, confidence_panel, gain_panel, innovation_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['UKF Filter (Up)', 'UKF Filter (Down)', 'UKF Filter (Neutral)', 'Uncertainty Band'], 
                      loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照线を追加
        panel_offset = 2 if show_volume else 1
        
        # 速度パネル
        axes[panel_offset].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 加速度パネル
        axes[panel_offset + 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 不確実性パネル（平均線）
        unc_mean = df['ukf_uncertainty'].mean()
        axes[panel_offset + 2].axhline(y=unc_mean, color='black', linestyle='--', alpha=0.3)
        
        # カルマンゲインパネル
        axes[panel_offset + 3].axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        axes[panel_offset + 3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_offset + 3].axhline(y=1, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        self._display_statistics(df)
        
        # 保存または表示
        if savefig:
            os.makedirs(os.path.dirname(savefig), exist_ok=True)
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def _display_statistics(self, df: pd.DataFrame) -> None:
        """統計情報を表示"""
        print(f"\n=== UKF統計情報 ===")
        
        # 基本統計
        valid_data = df.dropna(subset=['ukf_filtered', 'ukf_velocity', 'ukf_acceleration'])
        print(f"有効データ点数: {len(valid_data)}")
        
        # トレンド分析
        uptrend_count = (valid_data['ukf_velocity'] > 0).sum()
        downtrend_count = (valid_data['ukf_velocity'] < 0).sum()
        neutral_count = (valid_data['ukf_velocity'] == 0).sum()
        
        total_count = len(valid_data)
        print(f"上昇トレンド: {uptrend_count} ({uptrend_count/total_count*100:.1f}%)")
        print(f"下降トレンド: {downtrend_count} ({downtrend_count/total_count*100:.1f}%)")
        print(f"中立: {neutral_count} ({neutral_count/total_count*100:.1f}%)")
        
        # フィルタリング性能
        print(f"\n=== フィルタリング性能 ===")
        print(f"平均不確実性: {valid_data['ukf_uncertainty'].mean():.6f}")
        print(f"平均信頼度: {valid_data['ukf_confidence'].mean():.4f}")
        print(f"平均カルマンゲイン: {valid_data['ukf_kalman_gain'].mean():.4f}")
        
        # 速度・加速度統計
        print(f"\n=== 動力学統計 ===")
        print(f"速度 - 平均: {valid_data['ukf_velocity'].mean():.6f}, 標準偏差: {valid_data['ukf_velocity'].std():.6f}")
        print(f"加速度 - 平均: {valid_data['ukf_acceleration'].mean():.6f}, 標準偏差: {valid_data['ukf_acceleration'].std():.6f}")
        
        # 追跡誤差
        if 'close' in df.columns:
            tracking_error = np.sqrt(np.mean((valid_data['ukf_filtered'] - valid_data['close']) ** 2))
            print(f"\n追跡誤差 (RMSE): {tracking_error:.4f}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='無香料カルマンフィルター (UKF) チャート描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソース (close, hlc3, hl2, ohlc4)')
    parser.add_argument('--alpha', type=float, default=0.001, help='UKFアルファパラメータ')
    parser.add_argument('--beta', type=float, default=2.0, help='UKFベータパラメータ')
    parser.add_argument('--kappa', type=float, default=0.0, help='UKFカッパパラメータ')
    parser.add_argument('--process-noise', type=float, default=0.001, help='プロセスノイズスケール')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示にする')
    parser.add_argument('--no-adaptive', action='store_true', help='適応的ノイズ推定を無効にする')
    
    args = parser.parse_args()
    
    print("🚀 無香料カルマンフィルター (UKF) チャート描画開始\n")
    
    try:
        # 出力ディレクトリの作成
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        else:
            os.makedirs('examples/output', exist_ok=True)
        
        # チャートを作成
        chart = UKFChart()
        
        # データ読み込み
        chart.load_data_from_config(args.config)
        
        # UKF計算
        chart.calculate_ukf(
            src_type=args.src_type,
            alpha=args.alpha,
            beta=args.beta,
            kappa=args.kappa,
            process_noise_scale=args.process_noise,
            adaptive_noise=not args.no_adaptive
        )
        
        # チャート描画
        output_path = args.output or 'examples/output/ukf_chart.png'
        chart.plot(
            title=f"UKF Chart (α={args.alpha}, β={args.beta}, κ={args.kappa})",
            start_date=args.start,
            end_date=args.end,
            show_volume=not args.no_volume,
            savefig=output_path
        )
        
        print(f"\n🎉 チャート描画が完了しました！")
        if args.output:
            print(f"📊 出力ファイル: {args.output}")
        else:
            print(f"📊 出力ファイル: {output_path}")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 