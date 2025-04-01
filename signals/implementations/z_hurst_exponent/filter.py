#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.z_hurst_exponent import ZHurstExponent


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    hurst_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        hurst_values: Zハースト指数値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(hurst_values)
    signals = np.ones(length)  # デフォルトはトレンド相場 (1)
    
    for i in prange(length):
        if np.isnan(hurst_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif hurst_values[i] < threshold_values[i]:
            signals[i] = -1  # レンジ相場
    
    return signals


class ZHurstExponentSignal(BaseSignal, IFilterSignal):
    """
    Zハースト指数を使用したフィルターシグナル
    
    特徴:
    - サイクル効率比（CER）とドミナントサイクル検出を組み合わせて市場状態を判定
    - 動的しきい値でより正確な市場状態の検出が可能
    - 長期記憶特性に基づいてトレンド相場とレンジ相場を高精度に識別
    
    動作:
    - ハースト指数値が動的しきい値以上：トレンド相場 (1)
    - ハースト指数値が動的しきい値未満：レンジ相場 (-1)
    
    使用方法:
    - トレンド系/レンジ系ストラテジーの自動切り替え
    - エントリー条件の最適化
    - リスク管理の調整
    - 平均回帰とトレンドフォロー戦略の使い分け
    """
    
    def __init__(
        self,
        # 分析ウィンドウパラメータ
        max_window_dc_cycle_part: float = 0.75,
        max_window_dc_max_cycle: int = 144,
        max_window_dc_min_cycle: int = 8,
        max_window_dc_max_output: int = 200,
        max_window_dc_min_output: int = 50,
        
        min_window_dc_cycle_part: float = 0.5,
        min_window_dc_max_cycle: int = 55,
        min_window_dc_min_cycle: int = 5,
        min_window_dc_max_output: int = 50,
        min_window_dc_min_output: int = 20,
        
        # ラグパラメータ
        max_lag_ratio: float = 0.5,  # 最大ラグはウィンドウの何%か
        min_lag_ratio: float = 0.1,  # 最小ラグはウィンドウの何%か
        
        # ステップパラメータ
        max_step: int = 10,
        min_step: int = 2,
        
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'dudi_dce',
        lp_period: int = 10,
        hp_period: int = 48,
        cycle_part: float = 0.5,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.7,
        min_threshold: float = 0.55,
        
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            max_window_dc_cycle_part: 最大ウィンドウ用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.75）
            max_window_dc_max_cycle: 最大ウィンドウ用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_window_dc_min_cycle: 最大ウィンドウ用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 8）
            max_window_dc_max_output: 最大ウィンドウ用ドミナントサイクル計算用の最大出力値（デフォルト: 200）
            max_window_dc_min_output: 最大ウィンドウ用ドミナントサイクル計算用の最小出力値（デフォルト: 50）
            
            min_window_dc_cycle_part: 最小ウィンドウ用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            min_window_dc_max_cycle: 最小ウィンドウ用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_window_dc_min_cycle: 最小ウィンドウ用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_window_dc_max_output: 最小ウィンドウ用ドミナントサイクル計算用の最大出力値（デフォルト: 50）
            min_window_dc_min_output: 最小ウィンドウ用ドミナントサイクル計算用の最小出力値（デフォルト: 20）
            
            max_lag_ratio: 最大ラグとウィンドウの比率（デフォルト: 0.5）
            min_lag_ratio: 最小ラグとウィンドウの比率（デフォルト: 0.1）
            
            max_step: 最大ステップ（デフォルト: 10）
            min_step: 最小ステップ（デフォルト: 2）
            
            cycle_detector_type: サイクル検出器タイプ（デフォルト: 'dudi_dce'）
            lp_period: ローパスフィルターの期間（デフォルト: 10）
            hp_period: ハイパスフィルターの期間（デフォルト: 48）
            cycle_part: サイクル部分（デフォルト: 0.5）
            
            max_threshold: 最大しきい値（デフォルト: 0.7）
            min_threshold: 最小しきい値（デフォルト: 0.55）
            
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
        """
        # パラメータの設定
        params = {
            'max_window_dc_cycle_part': max_window_dc_cycle_part,
            'max_window_dc_max_cycle': max_window_dc_max_cycle,
            'max_window_dc_min_cycle': max_window_dc_min_cycle,
            'max_window_dc_max_output': max_window_dc_max_output,
            'max_window_dc_min_output': max_window_dc_min_output,
            'min_window_dc_cycle_part': min_window_dc_cycle_part,
            'min_window_dc_max_cycle': min_window_dc_max_cycle,
            'min_window_dc_min_cycle': min_window_dc_min_cycle,
            'min_window_dc_max_output': min_window_dc_max_output,
            'min_window_dc_min_output': min_window_dc_min_output,
            'max_lag_ratio': max_lag_ratio,
            'min_lag_ratio': min_lag_ratio,
            'max_step': max_step,
            'min_step': min_step,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'src_type': src_type
        }
        
        super().__init__(
            f"ZHurstExponent({cycle_detector_type}, {max_window_dc_max_output}, {min_window_dc_max_output})",
            params
        )
        
        # Zハースト指数インジケーターの初期化
        self._indicator = ZHurstExponent(
            # 分析ウィンドウパラメータ
            max_window_dc_cycle_part=max_window_dc_cycle_part,
            max_window_dc_max_cycle=max_window_dc_max_cycle,
            max_window_dc_min_cycle=max_window_dc_min_cycle,
            max_window_dc_max_output=max_window_dc_max_output,
            max_window_dc_min_output=max_window_dc_min_output,
            
            min_window_dc_cycle_part=min_window_dc_cycle_part,
            min_window_dc_max_cycle=min_window_dc_max_cycle,
            min_window_dc_min_cycle=min_window_dc_min_cycle,
            min_window_dc_max_output=min_window_dc_max_output,
            min_window_dc_min_output=min_window_dc_min_output,
            
            # ラグパラメータ
            max_lag_ratio=max_lag_ratio,
            min_lag_ratio=min_lag_ratio,
            
            # ステップパラメータ
            max_step=max_step,
            min_step=min_step,
            
            # サイクル効率比(CER)のパラメーター
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            # 動的しきい値のパラメータ
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            
            src_type=src_type
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
            
            # Zハースト指数の計算
            self._indicator.calculate(data)
            
            # ハースト指数値と動的しきい値の取得
            hurst_values = self._indicator._values
            threshold_values = self._indicator.get_adaptive_thresholds()
            
            # 計算が失敗した場合はNaNシグナルを返す
            if hurst_values is None or threshold_values is None:
                self._signals = np.full(len(data), np.nan)
                return self._signals
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(hurst_values, threshold_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZHurstExponentSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_filter_values(self) -> np.ndarray:
        """
        Zハースト指数値を取得する
        
        Returns:
            Zハースト指数値の配列
        """
        if hasattr(self._indicator, '_values') and self._indicator._values is not None:
            return self._indicator._values
        return np.array([])
    
    def get_threshold_values(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            動的しきい値の配列
        """
        if hasattr(self._indicator, '_result') and self._indicator._result is not None:
            return self._indicator._result.adaptive_thresholds
        return np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比（CER）を取得する
        
        Returns:
            サイクル効率比の配列
        """
        if hasattr(self._indicator, '_result') and self._indicator._result is not None:
            return self._indicator._result.er
        return np.array([])
    
    def get_trend_strength(self) -> np.ndarray:
        """
        トレンド強度を取得する
        
        Returns:
            トレンド強度の配列
        """
        if hasattr(self._indicator, '_result') and self._indicator._result is not None:
            return self._indicator._result.trend_strength
        return np.array([])
    
    def get_adaptive_windows(self) -> np.ndarray:
        """
        適応的な分析ウィンドウを取得する
        
        Returns:
            適応的な分析ウィンドウの配列
        """
        if hasattr(self._indicator, '_result') and self._indicator._result is not None:
            return self._indicator._result.adaptive_windows
        return np.array([])
    
    def get_adaptive_lags(self) -> tuple:
        """
        適応的なラグパラメータを取得する
        
        Returns:
            (最小ラグ, 最大ラグ)のタプル
        """
        if hasattr(self._indicator, '_result') and self._indicator._result is not None:
            return (
                self._indicator._result.adaptive_min_lags,
                self._indicator._result.adaptive_max_lags
            )
        return np.array([]), np.array([])
    
    def get_dominant_cycle(self) -> np.ndarray:
        """
        ドミナントサイクル値を取得する
        
        Returns:
            ドミナントサイクル値の配列
        """
        if hasattr(self._indicator, '_result') and self._indicator._result is not None:
            return self._indicator._result.dc_values
        return np.array([])
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._indicator, 'reset'):
            self._indicator.reset()
        self._signals = None
        self._data_hash = None 