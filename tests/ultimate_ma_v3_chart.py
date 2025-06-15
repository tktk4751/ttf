#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Optional, Tuple
import sys
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# UltimateMA V3インジケーター
from indicators.ultimate_ma_v3 import UltimateMAV3


class UltimateMAV3Chart:
    """
    UltimateMA V3を表示するローソク足チャートクラス
    
    表示内容:
    - メインパネル: ローソク足、UltimateMA V3ライン、シグナル
    - サブパネル1: 出来高（オプション）
    - サブパネル2: トレンド信号と信頼度
    - サブパネル3: 量子状態とMTF合意度
    - サブパネル4: フラクタル次元とエントロピー
    - サブパネル5: ボラティリティレジーム
    - サブパネル6: 各段階のフィルター結果
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_ma_v3 = None
        self.fig = None
        self.axes = None
    


    def load_binance_data_direct(self, symbol='BTC', market_type='spot', timeframe='4h', data_dir='data/binance'):
        """
        Binanceデータを直接読み込む
        
        Args:
            symbol: シンボル名 (BTC, ETH, etc.)
            market_type: 市場タイプ (spot, future)
            timeframe: 時間足 (1h, 4h, 1d, etc.)
            data_dir: データディレクトリのパス
        
        Returns:
            pd.DataFrame: OHLCVデータ
        """
        file_path = f"{data_dir}/{symbol}/{market_type}/{timeframe}/historical_data.csv"
        
        print(f"📂 データファイル読み込み中: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ データファイルが見つかりません: {file_path}")
            return None
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(file_path)
            
            # タイムスタンプをインデックスに設定
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # 必要なカラムが存在するか確認
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ 必要なカラムが不足しています: {missing_columns}")
                return None
            
            # データ型を数値に変換
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # NaNを除去
            df = df.dropna()
            
            self.data = df
            
            print(f"✅ データ読み込み成功: {symbol} {market_type} {timeframe}")
            print(f"📊 データ期間: {df.index.min()} - {df.index.max()}")
            print(f"📈 データ数: {len(df)}件")
            print(f"💰 価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return None

    def calculate_indicators(self,
                            super_smooth_period: int = 8,
                            zero_lag_period: int = 16,
                            realtime_window: int = 34,
                            quantum_window: int = 16,
                            fractal_window: int = 16,
                            entropy_window: int = 16,
                            src_type: str = 'hlc3',
                            slope_index: int = 2,
                            base_threshold: float = 0.002,
                            min_confidence: float = 0.15) -> None:
        """
        UltimateMA V3を計算する
        
        Args:
            super_smooth_period: スーパースムーザーフィルター期間
            zero_lag_period: ゼロラグEMA期間
            realtime_window: リアルタイムトレンド検出ウィンドウ
            quantum_window: 量子分析ウィンドウ
            fractal_window: フラクタル分析ウィンドウ
            entropy_window: エントロピー分析ウィンドウ
            src_type: 価格ソース
            slope_index: トレンド判定期間
            base_threshold: 基本閾値
            min_confidence: 最小信頼度
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()またはload_binance_data_direct()を先に実行してください。")
            
        print(f"\nUltimateMA V3を計算中...")
        print(f"設定: SS={super_smooth_period}, ZL={zero_lag_period}, RT={realtime_window}")
        print(f"      Quantum={quantum_window}, Fractal={fractal_window}, Entropy={entropy_window}")
        
        # UltimateMA V3インジケーターを初期化
        self.ultimate_ma_v3 = UltimateMAV3(
            super_smooth_period=super_smooth_period,
            zero_lag_period=zero_lag_period,
            realtime_window=realtime_window,
            quantum_window=quantum_window,
            fractal_window=fractal_window,
            entropy_window=entropy_window,
            src_type=src_type,
            slope_index=slope_index,
            base_threshold=base_threshold,
            min_confidence=min_confidence
        )
        
        # UltimateMA V3の計算
        print("計算を実行します...")
        result = self.ultimate_ma_v3.calculate(self.data)
        
        print(f"計算完了:")
        print(f"  - UltimateMA V3: {len(result.values)} ポイント")
        print(f"  - トレンド信号: {len(result.trend_signals)} ポイント")
        print(f"  - 信頼度: 平均={np.nanmean(result.trend_confidence):.3f}")
        print(f"  - 量子状態: 範囲=[{np.nanmin(result.quantum_state):.3f}, {np.nanmax(result.quantum_state):.3f}]")
        print(f"  - MTF合意度: 平均={np.nanmean(result.multi_timeframe_consensus):.3f}")
        
        # シグナル統計
        up_signals = np.sum(result.trend_signals == 1)
        down_signals = np.sum(result.trend_signals == -1)
        range_signals = np.sum(result.trend_signals == 0)
        total_signals = len(result.trend_signals)
        
        print(f"  - シグナル分布: 上昇={up_signals}({up_signals/total_signals*100:.1f}%), "
              f"下降={down_signals}({down_signals/total_signals*100:.1f}%), "
              f"レンジ={range_signals}({range_signals/total_signals*100:.1f}%)")
        
        # NaN値のチェック
        print(f"NaN値:")
        print(f"  - UltimateMA V3: {np.isnan(result.values).sum()}")
        print(f"  - トレンド信頼度: {np.isnan(result.trend_confidence).sum()}")
        print(f"  - 量子状態: {np.isnan(result.quantum_state).sum()}")
        print(f"  - フラクタル次元: {np.isnan(result.fractal_dimension).sum()}")
        print(f"  - エントロピー: {np.isnan(result.entropy_level).sum()}")
        
        # 現在のトレンド状態表示
        print(f"現在のトレンド: {result.current_trend} (信頼度: {result.current_confidence:.3f})")
        
        print("UltimateMA V3計算完了")
            
    def plot(self, 
            title: str = "UltimateMA V3 Analysis", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_signals: bool = True,
            show_filters: bool = True,
            figsize: Tuple[int, int] = (20, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            max_data_points: int = 2000) -> None:
        """
        ローソク足チャートとUltimateMA V3を描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_signals: シグナルマーカーを表示するか
            show_filters: フィルター段階を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            max_data_points: 最大データポイント数
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
            
        if self.ultimate_ma_v3 is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # データポイント数制限
        if len(df) > max_data_points:
            print(f"データポイント数が{max_data_points}を超えています。最新{max_data_points}件に制限します。")
            df = df.tail(max_data_points)
            
        # UltimateMA V3の結果を取得
        print("UltimateMA V3データを取得中...")
        result = self.ultimate_ma_v3._result
        
        if result is None:
            print("警告: UltimateMA V3の計算結果が空です。")
            return
        
        print(f"インジケーター結果のサイズ: {len(result.values)}")
        print(f"チャート用データのサイズ: {len(df)}")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'ultimate_ma_v3': result.values,
                'trend_signals': result.trend_signals,
                'trend_confidence': result.trend_confidence,
                'quantum_state': result.quantum_state,
                'mtf_consensus': result.multi_timeframe_consensus,
                'volatility_regime': result.volatility_regime,
                'fractal_dimension': result.fractal_dimension,
                'entropy_level': result.entropy_level,
                'kalman_values': result.kalman_values,
                'super_smooth_values': result.super_smooth_values,
                'zero_lag_values': result.zero_lag_values,
                'amplitude': result.amplitude,
                'realtime_trends': result.realtime_trends
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"期間: {df.index.min()} → {df.index.max()}")
        
        # データ有効性の確認
        def has_valid_data(series):
            if series is None or len(series) == 0:
                return False
            return len(series.dropna()) > 0 and not series.isna().all()
        
        print(f"\n有効データチェック:")
        data_validity = {}
        for col in ['ultimate_ma_v3', 'trend_confidence', 'quantum_state', 'mtf_consensus', 
                   'fractal_dimension', 'entropy_level', 'volatility_regime']:
            if col in df.columns:
                valid_count = len(df[col].dropna())
                total_count = len(df[col])
                data_validity[col] = has_valid_data(df[col])
                print(f"    {col}: {valid_count}/{total_count} 有効値")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # UltimateMA V3のメインライン
        if data_validity.get('ultimate_ma_v3', False):
            main_plots.append(mpf.make_addplot(df['ultimate_ma_v3'], color='blue', width=3, label='UltimateMA V3'))
        
        # フィルター段階の表示（オプション）
        if show_filters:
            if 'kalman_values' in df.columns and has_valid_data(df['kalman_values']):
                main_plots.append(mpf.make_addplot(df['kalman_values'], color='lightblue', width=1, alpha=0.5, label='Kalman'))
            if 'super_smooth_values' in df.columns and has_valid_data(df['super_smooth_values']):
                main_plots.append(mpf.make_addplot(df['super_smooth_values'], color='lightgreen', width=1, alpha=0.5, label='SuperSmooth'))
            if 'zero_lag_values' in df.columns and has_valid_data(df['zero_lag_values']):
                main_plots.append(mpf.make_addplot(df['zero_lag_values'], color='lightcoral', width=1, alpha=0.5, label='ZeroLag'))
        
        # シグナルマーカー（オプション）
        if show_signals and 'trend_signals' in df.columns:
            # ロングシグナル
            long_mask = df['trend_signals'] == 1
            if long_mask.any():
                long_signals_y = df.loc[long_mask, 'low'] * 0.995
                if len(long_signals_y) > 0:
                    long_plot_data = pd.Series(index=df.index, dtype=float)
                    long_plot_data.loc[long_mask] = long_signals_y
                    main_plots.append(mpf.make_addplot(
                        long_plot_data, type='scatter', markersize=120, 
                        marker='^', color='green', alpha=0.8, label='Long Signal'
                    ))
            
            # ショートシグナル
            short_mask = df['trend_signals'] == -1
            if short_mask.any():
                short_signals_y = df.loc[short_mask, 'high'] * 1.005
                if len(short_signals_y) > 0:
                    short_plot_data = pd.Series(index=df.index, dtype=float)
                    short_plot_data.loc[short_mask] = short_signals_y
                    main_plots.append(mpf.make_addplot(
                        short_plot_data, type='scatter', markersize=120, 
                        marker='v', color='red', alpha=0.8, label='Short Signal'
                    ))
        
        # サブパネル用のプロット
        sub_plots = []
        current_panel = 1 if show_volume else 0
        
        # トレンド信号と信頼度パネル
        current_panel += 1
        if data_validity.get('trend_confidence', False):
            confidence_panel = mpf.make_addplot(df['trend_confidence'], panel=current_panel, color='orange', width=2, 
                                              ylabel='Confidence', secondary_y=False, label='Confidence')
            sub_plots.append(confidence_panel)
        
        # トレンド信号をバーで表示
        if 'trend_signals' in df.columns and has_valid_data(df['trend_signals']):
            trend_panel = mpf.make_addplot(df['trend_signals'], panel=current_panel, color='purple', width=1.5, 
                                         secondary_y=True, label='Trend Signal')
            sub_plots.append(trend_panel)
        
        # 量子状態とMTF合意度パネル
        current_panel += 1
        if data_validity.get('quantum_state', False):
            quantum_panel = mpf.make_addplot(df['quantum_state'], panel=current_panel, color='purple', width=2, 
                                           ylabel='Quantum/MTF', secondary_y=False, label='Quantum')
            sub_plots.append(quantum_panel)
        
        if data_validity.get('mtf_consensus', False):
            mtf_panel = mpf.make_addplot(df['mtf_consensus'], panel=current_panel, color='blue', width=1.5, 
                                       secondary_y=True, label='MTF Consensus')
            sub_plots.append(mtf_panel)
        
        # フラクタル次元とエントロピーパネル
        current_panel += 1
        if data_validity.get('fractal_dimension', False):
            fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=current_panel, color='green', width=2, 
                                           ylabel='Fractal/Entropy', secondary_y=False, label='Fractal')
            sub_plots.append(fractal_panel)
        
        if data_validity.get('entropy_level', False):
            entropy_panel = mpf.make_addplot(df['entropy_level'], panel=current_panel, color='red', width=1.5, 
                                           secondary_y=True, label='Entropy')
            sub_plots.append(entropy_panel)
        
        # ボラティリティレジームパネル
        current_panel += 1
        if data_validity.get('volatility_regime', False):
            vol_regime_panel = mpf.make_addplot(df['volatility_regime'], panel=current_panel, color='brown', width=2, 
                                              ylabel='Vol Regime', secondary_y=False, label='Vol Regime')
            sub_plots.append(vol_regime_panel)
        
        # 何もプロットするものがない場合の警告
        if not main_plots and not sub_plots:
            print("警告: 表示可能なデータがありません。計算結果を確認してください。")
            return
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            warn_too_much_data=len(df) + 1000
        )
        
        # パネル数の動的計算
        total_panels = 1  # メインパネル
        if show_volume:
            total_panels += 1
        total_panels += 4  # 4つのサブパネル
        
        # パネル構成の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (6, 1, 2, 2, 2, 1.5)  # メイン, 出来高, 信頼度, 量子/MTF, フラクタル/エントロピー, ボラティリティ
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (6, 2, 2, 2, 1.5)  # メイン, 信頼度, 量子/MTF, フラクタル/エントロピー, ボラティリティ
        
        # すべてのプロットを結合
        all_plots = main_plots + sub_plots
        if all_plots:
            kwargs['addplot'] = all_plots
        
        try:
            # プロット実行
            fig, axes = mpf.plot(df, **kwargs)
            
            # 凡例の追加（メインパネル）
            if main_plots:
                legend_labels = ['UltimateMA V3']
                if show_filters:
                    legend_labels.extend(['Kalman', 'SuperSmooth', 'ZeroLag'])
                if show_signals:
                    if (df['trend_signals'] == 1).any():
                        legend_labels.append('Long Signal')
                    if (df['trend_signals'] == -1).any():
                        legend_labels.append('Short Signal')
                
                if legend_labels:
                    axes[0].legend(legend_labels, loc='upper left', fontsize=8)
            
            self.fig = fig
            self.axes = axes
            
            # 参照線の追加
            panel_offset = 1 if show_volume else 0
            
            # 信頼度パネルの参照線
            confidence_panel_idx = 1 + panel_offset
            if confidence_panel_idx < len(axes):
                axes[confidence_panel_idx].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='High Conf')
                axes[confidence_panel_idx].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Ultra Conf')
                axes[confidence_panel_idx].set_ylim(0, 1)
                
                # 右軸（トレンド信号）の参照線
                ax_right = axes[confidence_panel_idx].twinx() if hasattr(axes[confidence_panel_idx], 'twinx') else None
                if ax_right:
                    ax_right.axhline(y=1, color='green', linestyle='--', alpha=0.3)
                    ax_right.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    ax_right.axhline(y=-1, color='red', linestyle='--', alpha=0.3)
                    ax_right.set_ylim(-1.5, 1.5)
            
            # 量子状態パネルの参照線
            quantum_panel_idx = 2 + panel_offset
            if quantum_panel_idx < len(axes):
                axes[quantum_panel_idx].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # 右軸（MTF合意度）の参照線
                ax_right = axes[quantum_panel_idx].twinx() if hasattr(axes[quantum_panel_idx], 'twinx') else None
                if ax_right:
                    ax_right.axhline(y=0.8, color='blue', linestyle='--', alpha=0.5, label='Strong Consensus')
                    ax_right.set_ylim(0, 1)
            
            # フラクタル次元パネルの参照線
            fractal_panel_idx = 3 + panel_offset
            if fractal_panel_idx < len(axes):
                axes[fractal_panel_idx].axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Smooth Trend')
                axes[fractal_panel_idx].set_ylim(1, 2)
                
                # 右軸（エントロピー）の参照線
                ax_right = axes[fractal_panel_idx].twinx() if hasattr(axes[fractal_panel_idx], 'twinx') else None
                if ax_right:
                    ax_right.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Mid Entropy')
                    ax_right.set_ylim(0, 1)
            
            # ボラティリティレジームパネルの参照線
            vol_panel_idx = 4 + panel_offset
            if vol_panel_idx < len(axes):
                axes[vol_panel_idx].axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Low Vol')
                axes[vol_panel_idx].axhline(y=1, color='gray', linestyle='-', alpha=0.3, label='Normal')
                axes[vol_panel_idx].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='High Vol')
                axes[vol_panel_idx].set_ylim(-0.5, 2.5)
            
            # 保存または表示
            if savefig:
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"チャートを保存しました: {savefig}")
            else:
                try:
                    plt.tight_layout()
                except:
                    plt.subplots_adjust(hspace=0.4, wspace=0.1)
                plt.show()
                
        except Exception as e:
            print(f"チャート描画中にエラーが発生しました: {str(e)}")
            
            # 基本的なチャートのみ表示を試行
            try:
                print("基本チャートのみで再試行...")
                basic_kwargs = dict(
                    type='candle',
                    figsize=figsize,
                    title=title,
                    style=style,
                    datetime_format='%Y-%m-%d',
                    xrotation=45,
                    returnfig=True,
                    warn_too_much_data=len(df) + 1000
                )
                
                if show_volume:
                    basic_kwargs['volume'] = True
                
                if main_plots:
                    basic_kwargs['addplot'] = main_plots
                
                fig, axes = mpf.plot(df, **basic_kwargs)
                self.fig = fig
                self.axes = axes
                
                if savefig:
                    try:
                        plt.tight_layout()
                    except:
                        pass
                    plt.savefig(savefig, dpi=300, bbox_inches='tight')
                    print(f"基本チャートを保存しました: {savefig}")
                else:
                    try:
                        plt.tight_layout()
                    except:
                        plt.subplots_adjust(hspace=0.3, wspace=0.1)
                    plt.show()
                    
            except Exception as e2:
                print(f"基本チャート描画も失敗しました: {str(e2)}")
                raise e

    def print_statistics(self) -> None:
        """
        UltimateMA V3の統計情報を表示
        """
        if self.ultimate_ma_v3 is None:
            print("インジケーターが計算されていません。")
            return
        
        result = self.ultimate_ma_v3._result
        if result is None:
            print("計算結果がありません。")
            return
        
        print("\n=== UltimateMA V3 統計情報 ===")
        print(f"現在のトレンド: {result.current_trend} (信頼度: {result.current_confidence:.3f})")
        
        # 最新値
        if len(result.values) > 0:
            print(f"\n最新値:")
            print(f"  - UltimateMA V3: {result.values[-1]:.4f}")
            print(f"  - 量子状態: {result.quantum_state[-1]:.4f}")
            print(f"  - MTF合意度: {result.multi_timeframe_consensus[-1]:.3f}")
            print(f"  - フラクタル次元: {result.fractal_dimension[-1]:.3f}")
            print(f"  - エントロピー: {result.entropy_level[-1]:.3f}")
            print(f"  - ボラティリティレジーム: {result.volatility_regime[-1]}")
        
        # シグナル統計
        up_signals = np.sum(result.trend_signals == 1)
        down_signals = np.sum(result.trend_signals == -1)
        range_signals = np.sum(result.trend_signals == 0)
        total_signals = len(result.trend_signals)
        
        print(f"\nシグナル統計:")
        print(f"  - 上昇シグナル: {up_signals}回 ({up_signals/total_signals*100:.1f}%)")
        print(f"  - 下降シグナル: {down_signals}回 ({down_signals/total_signals*100:.1f}%)")
        print(f"  - レンジシグナル: {range_signals}回 ({range_signals/total_signals*100:.1f}%)")
        
        # 信頼度統計
        valid_confidence = result.trend_confidence[result.trend_confidence > 0]
        if len(valid_confidence) > 0:
            print(f"\n信頼度統計:")
            print(f"  - 平均信頼度: {np.mean(valid_confidence):.3f}")
            print(f"  - 最大信頼度: {np.max(valid_confidence):.3f}")
            print(f"  - 高信頼度(>0.5): {np.sum(valid_confidence > 0.5)}回 ({np.sum(valid_confidence > 0.5)/len(valid_confidence)*100:.1f}%)")
        
        # 量子分析統計
        print(f"\n量子分析統計:")
        print(f"  - 量子状態: 平均={np.nanmean(result.quantum_state):.3f}, "
              f"範囲=[{np.nanmin(result.quantum_state):.3f}, {np.nanmax(result.quantum_state):.3f}]")
        print(f"  - MTF合意度: 平均={np.nanmean(result.multi_timeframe_consensus):.3f}")
        print(f"  - フラクタル次元: 平均={np.nanmean(result.fractal_dimension):.3f}")
        print(f"  - エントロピー: 平均={np.nanmean(result.entropy_level):.3f}")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='UltimateMA V3の描画')
    parser.add_argument('--config', '-c', type=str, help='設定ファイルのパス')
    parser.add_argument('--symbol', '-s', type=str, default='BTC', help='シンボル名 (BTC, ETH, etc.)')
    parser.add_argument('--market', '-m', type=str, default='spot', help='市場タイプ (spot, future)')
    parser.add_argument('--timeframe', '-t', type=str, default='4h', help='時間足 (1h, 4h, 1d, etc.)')
    parser.add_argument('--data-dir', type=str, default='data/binance', help='データディレクトリ')
    parser.add_argument('--start', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--super-smooth', type=int, default=8, help='スーパースムーザー期間')
    parser.add_argument('--zero-lag', type=int, default=16, help='ゼロラグEMA期間')
    parser.add_argument('--realtime', type=int, default=34, help='リアルタイムウィンドウ')
    parser.add_argument('--quantum', type=int, default=16, help='量子分析ウィンドウ')
    parser.add_argument('--fractal', type=int, default=16, help='フラクタル分析ウィンドウ')
    parser.add_argument('--entropy', type=int, default=16, help='エントロピー分析ウィンドウ')
    parser.add_argument('--threshold', type=float, default=0.002, help='基本閾値')
    parser.add_argument('--confidence', type=float, default=0.15, help='最小信頼度')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-signals', action='store_true', help='シグナルマーカーを非表示')
    parser.add_argument('--no-filters', action='store_true', help='フィルター段階を非表示')
    parser.add_argument('--stats', action='store_true', help='統計情報を表示')
    parser.add_argument('--max-points', type=int, default=2000, help='最大データポイント数')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateMAV3Chart()
    
    # データ読み込み
    if args.config:
        chart.load_data_from_config(args.config)
    else:
        chart.load_binance_data_direct(
            symbol=args.symbol,
            market_type=args.market,
            timeframe=args.timeframe,
            data_dir=args.data_dir
        )
    
    # インジケーター計算
    chart.calculate_indicators(
        super_smooth_period=args.super_smooth,
        zero_lag_period=args.zero_lag,
        realtime_window=args.realtime,
        quantum_window=args.quantum,
        fractal_window=args.fractal,
        entropy_window=args.entropy,
        base_threshold=args.threshold,
        min_confidence=args.confidence
    )
    
    # 統計情報表示（オプション）
    if args.stats:
        chart.print_statistics()
    
    # チャート描画
    chart.plot(
        title=f"UltimateMA V3 Analysis - {args.symbol}",
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_signals=not args.no_signals,
        show_filters=not args.no_filters,
        savefig=args.output,
        max_data_points=args.max_points
    )


if __name__ == "__main__":
    main() 