#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.trend_filter.x_choppiness import XChoppiness


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    trend_signal: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        trend_signal: X-Choppinessのtrend_signal配列（1=トレンド、-1=レンジ）
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(trend_signal)
    signals = np.ones(length)  # デフォルトはトレンド相場 (1)
    
    for i in prange(length):
        if np.isnan(trend_signal[i]):
            signals[i] = np.nan
        elif trend_signal[i] < 0:  # レンジ相場の場合
            signals[i] = -1
        else:  # trend_signal[i] > 0 の場合（トレンド相場）
            signals[i] = 1
    
    return signals


class XChoppinessFilterSignal(BaseSignal, IFilterSignal):
    """
    X-Choppinessを使用したフィルターシグナル
    
    特徴:
    - STRベースの改良チョピネスインデックス（X-Choppiness）を使用
    - ATRの代わりにSTRを使用することでよりスムーズな測定
    - 値を反転済み（高い値=トレンド、低い値=レンジ）
    - ミッドラインによる自動的なトレンド/レンジ判定
    - 0-1の値範囲で直感的な解釈が可能
    
    動作:
    - X-Choppiness > ミッドライン：トレンド相場 (1)
    - X-Choppiness < ミッドライン：レンジ相場 (-1)
    
    使用方法:
    - トレンド系戦略のフィルターとして
    - レンジ系戦略での逆張りフィルターとして  
    - 市場状態の判定とリスク管理の調整
    """
    
    def __init__(
        self,
        # X-Choppinessパラメータ（内部STR実装対応版）
        period: int = 14,
        midline_period: int = 100,
        # 平滑化オプション
        use_smoothing: bool = True,
        smoother_type: str = 'super_smoother',
        smoother_period: int = 8,
        smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ（オプション）
        use_dynamic_period: bool = False,
        detector_type: str = 'phac_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        # カルマンフィルターパラメータ（オプション）
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'unscented',
        kalman_process_noise: float = 0.01,
        kalman_observation_noise: float = 0.001,
        # パーセンタイル分析パラメータ（オプション）
        enable_percentile_analysis: bool = True,
        percentile_lookback_period: int = 50,
        percentile_low_threshold: float = 0.25,
        percentile_high_threshold: float = 0.75
    ):
        """
        コンストラクタ（内部STR実装対応版）
        
        Args:
            period: X-Choppiness計算期間（デフォルト: 14）
            midline_period: ミッドライン計算期間（デフォルト: 100）
            use_smoothing: 平滑化を使用するか（デフォルト: True）
            smoother_type: スムーサータイプ（デフォルト: 'super_smoother'）
            smoother_period: スムーサー期間（デフォルト: 8）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            use_dynamic_period: 動的期間適応を使用するか（デフォルト: False）
            detector_type: サイクル検出器タイプ（デフォルト: 'phac_e'）
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 144）
            cycle_part: サイクル部分（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 144）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 55）
            min_output: 最小出力値（デフォルト: 5）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: カルマンフィルタープロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: カルマンフィルター観測ノイズ（デフォルト: 0.001）
            enable_percentile_analysis: パーセンタイル分析を有効にするか（デフォルト: True）
            percentile_lookback_period: パーセンタイル分析のルックバック期間（デフォルト: 50）
            percentile_low_threshold: パーセンタイル分析の低閾値（デフォルト: 0.25）
            percentile_high_threshold: パーセンタイル分析の高閾値（デフォルト: 0.75）
        """
        # パラメータの設定（内部STR実装対応版）
        params = {
            'period': period,
            'midline_period': midline_period,
            'use_smoothing': use_smoothing,
            'smoother_type': smoother_type,
            'smoother_period': smoother_period,
            'smoother_src_type': smoother_src_type,
            'use_dynamic_period': use_dynamic_period,
            'detector_type': detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'enable_percentile_analysis': enable_percentile_analysis,
            'percentile_lookback_period': percentile_lookback_period,
            'percentile_low_threshold': percentile_low_threshold,
            'percentile_high_threshold': percentile_high_threshold
        }
        
        # シグナル名の構築（内部STR実装対応版）
        signal_name = f"XChoppinessFilter(period={period}, midline={midline_period}"
        if use_dynamic_period:
            signal_name += f", dynamic={detector_type}"
        if use_kalman_filter:
            signal_name += f", kalman={kalman_filter_type}"
        if use_smoothing:
            signal_name += f", smooth={smoother_type}({smoother_period})"
        signal_name += ")"
        
        super().__init__(signal_name, params)
        
        # X-Choppinessインジケーターの初期化（内部STR実装対応版）
        self._filter = XChoppiness(
            period=period,
            midline_period=midline_period,
            use_smoothing=use_smoothing,
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            smoother_src_type=smoother_src_type,
            use_dynamic_period=use_dynamic_period,
            detector_type=detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # 結果キャッシュ
        self._signals = None
        self._data_hash = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = f"{hash(frozenset(self._params.items()))}"
        
        return f"{data_hash}_{param_str}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
            elif data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # X-Choppinessの計算
            filter_result = self._filter.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if filter_result is None or filter_result.trend_signal is None or len(filter_result.trend_signal) == 0:
                self._signals = np.full(len(data), np.nan)
                return self._signals
                
            # trend_signalの取得（1=トレンド、-1=レンジ）
            trend_signal = filter_result.trend_signal
            
            # シグナルの生成（高速化版）
            # X-Choppinessのtrend_signalをそのまま使用
            signals = generate_signals_numba(trend_signal)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"XChoppinessFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_x_choppiness_values(self) -> np.ndarray:
        """
        X-Choppiness値を取得する
        
        Returns:
            X-Choppiness値の配列（0-1の範囲、高い値=トレンド）
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.values is not None:
                return result.values.copy()
        except Exception:
            pass
        return np.array([])
    
    def get_raw_choppiness_values(self) -> np.ndarray:
        """
        生のチョピネス値を取得する
        
        Returns:
            生のチョピネス値の配列
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.raw_choppiness is not None:
                return result.raw_choppiness.copy()
        except Exception:
            pass
        return np.array([])
    
    def get_smoothed_choppiness_values(self) -> np.ndarray:
        """
        平滑化されたチョピネス値を取得する
        
        Returns:
            平滑化されたチョピネス値の配列
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.smoothed_choppiness is not None:
                return result.smoothed_choppiness.copy()
        except Exception:
            pass
        return np.array([])
    
    def get_midline_values(self) -> np.ndarray:
        """
        ミッドライン値を取得する
        
        Returns:
            ミッドライン値の配列
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.midline is not None:
                return result.midline.copy()
        except Exception:
            pass
        return np.array([])
    
    def get_trend_signal_values(self) -> np.ndarray:
        """
        トレンド信号値を取得する
        
        Returns:
            トレンド信号の配列 (1=トレンド、-1=レンジ)
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.trend_signal is not None:
                return result.trend_signal.copy()
        except Exception:
            pass
        return np.array([])
    
    def get_str_values(self) -> np.ndarray:
        """
        STR値を取得する
        
        Returns:
            STR値の配列
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.str_values is not None:
                return result.str_values.copy()
        except Exception:
            pass
        return np.array([])
    
    def get_percentiles(self) -> Optional[np.ndarray]:
        """
        パーセンタイル値を取得する
        
        Returns:
            パーセンタイル値の配列（有効な場合）
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.percentiles is not None:
                return result.percentiles.copy()
        except Exception:
            pass
        return None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """
        トレンド状態を取得する
        
        Returns:
            トレンド状態の配列 (-1=レンジ、0=中、1=トレンド)（有効な場合）
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.trend_state is not None:
                return result.trend_state.copy()
        except Exception:
            pass
        return None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """
        トレンド強度を取得する
        
        Returns:
            トレンド強度の配列 (0-1の範囲)（有効な場合）
        """
        try:
            result = self._filter._get_latest_result()
            if result and result.trend_intensity is not None:
                return result.trend_intensity.copy()
        except Exception:
            pass
        return None
    
    def get_advanced_metrics(self) -> Dict[str, Any]:
        """
        高度なメトリクスを取得する
        
        Returns:
            X-Choppiness関連の全メトリクス
        """
        try:
            metrics = {
                'x_choppiness_values': self.get_x_choppiness_values(),
                'raw_choppiness_values': self.get_raw_choppiness_values(),
                'smoothed_choppiness_values': self.get_smoothed_choppiness_values(),
                'midline_values': self.get_midline_values(),
                'trend_signal_values': self.get_trend_signal_values(),
                'str_values': self.get_str_values(),
                'percentiles': self.get_percentiles(),
                'trend_state': self.get_trend_state(),
                'trend_intensity': self.get_trend_intensity(),
                'filter_info': self._filter.get_indicator_info()
            }
            
            # パーセンタイル分析サマリーを追加
            try:
                percentile_summary = self._filter.get_percentile_analysis_summary()
                if percentile_summary:
                    metrics['percentile_analysis_summary'] = percentile_summary
            except Exception:
                pass
            
            return metrics
            
        except Exception as e:
            print(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._filter, 'reset'):
            self._filter.reset()
        self._signals = None
        self._data_hash = None


# 便利関数
def create_x_choppiness_filter_signal(
    period: int = 14,
    midline_period: int = 100,
    use_smoothing: bool = True,
    smoother_type: str = 'super_smoother',
    smoother_period: int = 8,
    use_dynamic_period: bool = False,
    use_kalman_filter: bool = False,
    **kwargs
) -> XChoppinessFilterSignal:
    """
    X-Choppinessフィルターシグナルを作成する便利関数（内部STR実装対応版）
    
    Args:
        period: X-Choppiness計算期間（デフォルト: 14）
        midline_period: ミッドライン計算期間（デフォルト: 100）
        use_smoothing: 平滑化を使用するか（デフォルト: True）
        smoother_type: スムーサータイプ（デフォルト: 'super_smoother'）
        smoother_period: スムーサー期間（デフォルト: 8）
        use_dynamic_period: 動的期間適応を使用するか（デフォルト: False）
        use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
        **kwargs: その他のパラメータ
        
    Returns:
        XChoppinessFilterSignal: 設定済みのフィルターシグナルインスタンス
    """
    return XChoppinessFilterSignal(
        period=period,
        midline_period=midline_period,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        smoother_period=smoother_period,
        use_dynamic_period=use_dynamic_period,
        use_kalman_filter=use_kalman_filter,
        **kwargs
    )


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== X-Choppiness フィルターシグナルのテスト ===")
    
    # テストデータ生成（トレンドとレンジが混在）
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # トレンド相場
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.015)
        else:  # レンジ相場
            change = np.random.normal(0, 0.006)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本版X-Choppinessフィルターのテスト（内部STR実装版）
    print("\n=== 基本版X-Choppinessフィルターのテスト（内部STR実装版） ===")
    filter_signal = XChoppinessFilterSignal(
        period=14,
        midline_period=50,
        use_smoothing=True,
        smoother_type='frama'
    )
    
    signals = filter_signal.generate(df)
    x_chop_values = filter_signal.get_x_choppiness_values()
    midline_values = filter_signal.get_midline_values()
    trend_signals = filter_signal.get_trend_signal_values()
    
    valid_count = np.sum(~np.isnan(signals))
    trend_ratio = np.sum(signals == 1) / valid_count if valid_count > 0 else 0
    range_ratio = np.sum(signals == -1) / valid_count if valid_count > 0 else 0
    
    print(f"有効シグナル数: {valid_count}/{len(df)}")
    print(f"トレンド判定比率: {trend_ratio:.2%}")
    print(f"レンジ判定比率: {range_ratio:.2%}")
    print(f"平均X-Choppiness: {np.nanmean(x_chop_values):.4f}")
    print(f"平均ミッドライン: {np.nanmean(midline_values):.4f}")
    
    # 高度なメトリクスのテスト
    print("\n=== 高度なメトリクスのテスト ===")
    advanced_metrics = filter_signal.get_advanced_metrics()
    
    print(f"利用可能なメトリクス: {list(advanced_metrics.keys())}")
    
    # STR値の統計
    str_values = advanced_metrics.get('str_values', np.array([]))
    if len(str_values) > 0:
        print(f"平均STR値: {np.nanmean(str_values):.4f}")
    
    # パーセンタイル分析の結果
    percentile_summary = advanced_metrics.get('percentile_analysis_summary', {})
    if percentile_summary:
        print(f"パーセンタイル分析: {percentile_summary}")
    
    print("\n=== テスト完了 ===")