#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.cz_channel import CZChannel


@njit(fastmath=True, parallel=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, lookback: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定（並列処理化）
    for i in prange(lookback, length):
        # 終値とバンドの値が有効かチェック
        if np.isnan(close[i]) or np.isnan(upper[i-lookback]) or np.isnan(lower[i-lookback]):
            signals[i] = 0
            continue
            
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-lookback]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-lookback]:
            signals[i] = -1
    
    return signals


class CZChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    CZチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - サイクル効率比（CER）に基づく動的な適応性
    - 平滑化アルゴリズム（ALMAまたはハイパースムーサー）を使用したCATR
    - ボラティリティに応じた最適なバンド幅
    - トレンドの強さに合わせた自動調整
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 100,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 120,       # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_lp_period: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用LPピリオド
        zma_max_dc_hp_period: int = 55,         # ZMA: 最大期間用ドミナントサイクル計算用HPピリオド
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_lp_period: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用LPピリオド
        zma_min_dc_hp_period: int = 34,         # ZMA: 最小期間用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.5,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 89,
        zma_slow_max_dc_min_output: int = 22,
        zma_slow_max_dc_lp_period: int = 5,      # ZMA: Slow最大用ドミナントサイクル計算用LPピリオド
        zma_slow_max_dc_hp_period: int = 55,     # ZMA: Slow最大用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.5,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 5,
        zma_slow_min_dc_max_output: int = 21,
        zma_slow_min_dc_min_output: int = 8,
        zma_slow_min_dc_lp_period: int = 5,      # ZMA: Slow最小用ドミナントサイクル計算用LPピリオド
        zma_slow_min_dc_hp_period: int = 34,     # ZMA: Slow最小用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.5,
        zma_fast_max_dc_max_cycle: int = 55,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 3,
        zma_fast_max_dc_lp_period: int = 5,      # ZMA: Fast最大用ドミナントサイクル計算用LPピリオド
        zma_fast_max_dc_hp_period: int = 21,     # ZMA: Fast最大用ドミナントサイクル計算用HPピリオド
        
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """
        初期化
        
        Args:
            detector_type: 検出器タイプ（ZMAとCATRに使用）
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積（デフォルト）
                - 'dft': 離散フーリエ変換
            cer_detector_type: CER用の検出器タイプ（指定しない場合はdetector_typeと同じ）
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 55）
            cycle_part: サイクル部分の倍率（デフォルト: 0.7）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 6.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            # その他のパラメータは各コンポーネントのドキュメントを参照
        """
        super().__init__(
            f"CZChannelBreakoutEntrySignal({detector_type}, {max_max_multiplier}, {min_min_multiplier}, {band_lookback})"
        )
        
        # CERの検出器タイプが指定されていない場合、detector_typeと同じ値を使用
        if cer_detector_type is None:
            cer_detector_type = detector_type
            
        # パラメータの保存
        self._params = {
            # 基本パラメータ
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # ZMA用パラメータ
            'zma_max_dc_cycle_part': zma_max_dc_cycle_part,
            'zma_max_dc_max_cycle': zma_max_dc_max_cycle,
            'zma_max_dc_min_cycle': zma_max_dc_min_cycle,
            'zma_max_dc_max_output': zma_max_dc_max_output,
            'zma_max_dc_min_output': zma_max_dc_min_output,
            'zma_max_dc_lp_period': zma_max_dc_lp_period,
            'zma_max_dc_hp_period': zma_max_dc_hp_period,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_min_dc_lp_period': zma_min_dc_lp_period,
            'zma_min_dc_hp_period': zma_min_dc_hp_period,
            'zma_slow_max_dc_cycle_part': zma_slow_max_dc_cycle_part,
            'zma_slow_max_dc_max_cycle': zma_slow_max_dc_max_cycle,
            'zma_slow_max_dc_min_cycle': zma_slow_max_dc_min_cycle,
            'zma_slow_max_dc_max_output': zma_slow_max_dc_max_output,
            'zma_slow_max_dc_min_output': zma_slow_max_dc_min_output,
            'zma_slow_max_dc_lp_period': zma_slow_max_dc_lp_period,
            'zma_slow_max_dc_hp_period': zma_slow_max_dc_hp_period,
            'zma_slow_min_dc_cycle_part': zma_slow_min_dc_cycle_part,
            'zma_slow_min_dc_max_cycle': zma_slow_min_dc_max_cycle,
            'zma_slow_min_dc_min_cycle': zma_slow_min_dc_min_cycle,
            'zma_slow_min_dc_max_output': zma_slow_min_dc_max_output,
            'zma_slow_min_dc_min_output': zma_slow_min_dc_min_output,
            'zma_slow_min_dc_lp_period': zma_slow_min_dc_lp_period,
            'zma_slow_min_dc_hp_period': zma_slow_min_dc_hp_period,
            'zma_fast_max_dc_cycle_part': zma_fast_max_dc_cycle_part,
            'zma_fast_max_dc_max_cycle': zma_fast_max_dc_max_cycle,
            'zma_fast_max_dc_min_cycle': zma_fast_max_dc_min_cycle,
            'zma_fast_max_dc_max_output': zma_fast_max_dc_max_output,
            'zma_fast_max_dc_min_output': zma_fast_max_dc_min_output,
            'zma_fast_max_dc_lp_period': zma_fast_max_dc_lp_period,
            'zma_fast_max_dc_hp_period': zma_fast_max_dc_hp_period,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # CATR用パラメータ
            'catr_detector_type': catr_detector_type,
            'catr_cycle_part': catr_cycle_part,
            'catr_lp_period': catr_lp_period,
            'catr_hp_period': catr_hp_period,
            'catr_max_cycle': catr_max_cycle,
            'catr_min_cycle': catr_min_cycle,
            'catr_max_output': catr_max_output,
            'catr_min_output': catr_min_output,
            'catr_smoother_type': catr_smoother_type
        }
            
        # CZチャネルの初期化
        self.cz_channel = CZChannel(
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            # 動的乗数の範囲パラメータ
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # ZMA用パラメータ - すべて
            zma_max_dc_cycle_part=zma_max_dc_cycle_part,
            zma_max_dc_max_cycle=zma_max_dc_max_cycle,
            zma_max_dc_min_cycle=zma_max_dc_min_cycle,
            zma_max_dc_max_output=zma_max_dc_max_output,
            zma_max_dc_min_output=zma_max_dc_min_output,
            zma_max_dc_lp_period=zma_max_dc_lp_period,
            zma_max_dc_hp_period=zma_max_dc_hp_period,
            
            zma_min_dc_cycle_part=zma_min_dc_cycle_part,
            zma_min_dc_max_cycle=zma_min_dc_max_cycle,
            zma_min_dc_min_cycle=zma_min_dc_min_cycle,
            zma_min_dc_max_output=zma_min_dc_max_output,
            zma_min_dc_min_output=zma_min_dc_min_output,
            zma_min_dc_lp_period=zma_min_dc_lp_period,
            zma_min_dc_hp_period=zma_min_dc_hp_period,
            
            # 動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part=zma_slow_max_dc_cycle_part,
            zma_slow_max_dc_max_cycle=zma_slow_max_dc_max_cycle,
            zma_slow_max_dc_min_cycle=zma_slow_max_dc_min_cycle,
            zma_slow_max_dc_max_output=zma_slow_max_dc_max_output,
            zma_slow_max_dc_min_output=zma_slow_max_dc_min_output,
            zma_slow_max_dc_lp_period=zma_slow_max_dc_lp_period,
            zma_slow_max_dc_hp_period=zma_slow_max_dc_hp_period,
            
            # 動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part=zma_slow_min_dc_cycle_part,
            zma_slow_min_dc_max_cycle=zma_slow_min_dc_max_cycle,
            zma_slow_min_dc_min_cycle=zma_slow_min_dc_min_cycle,
            zma_slow_min_dc_max_output=zma_slow_min_dc_max_output,
            zma_slow_min_dc_min_output=zma_slow_min_dc_min_output,
            zma_slow_min_dc_lp_period=zma_slow_min_dc_lp_period,
            zma_slow_min_dc_hp_period=zma_slow_min_dc_hp_period,
            
            # 動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part=zma_fast_max_dc_cycle_part,
            zma_fast_max_dc_max_cycle=zma_fast_max_dc_max_cycle,
            zma_fast_max_dc_min_cycle=zma_fast_max_dc_min_cycle,
            zma_fast_max_dc_max_output=zma_fast_max_dc_max_output,
            zma_fast_max_dc_min_output=zma_fast_max_dc_min_output,
            zma_fast_max_dc_lp_period=zma_fast_max_dc_lp_period,
            zma_fast_max_dc_hp_period=zma_fast_max_dc_hp_period,
            
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
            # CATR用パラメータ - すべて
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # 参照期間の設定
        self.band_lookback = band_lookback
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]の4列が必要
        
        Returns:
            np.ndarray: シグナルの配列（1: ロング、-1: ショート、0: シグナルなし）
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
                df = data
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            
            # Zチャネルの計算
            z_channel = self.cz_channel.calculate(df)
            
            # バンドの取得
            middle, upper, lower = z_channel.middle, z_channel.upper, z_channel.lower
            
            # ブレイクアウトシグナルの計算
            signals = calculate_breakout_signals(
                close=df['close'].values,
                upper=upper,
                lower=lower,
                lookback=self.band_lookback
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            self.logger.error(f"CZChannelBreakoutEntrySignal計算中にエラー: {str(e)}")
            if data is not None:
                return np.zeros(len(data), dtype=np.int8)
            return np.array([], dtype=np.int8)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CZチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
                df = data
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                    
            result = self.cz_channel.calculate(df)
            return result.middle, result.upper, result.lower
            
        return None, None, None
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
                df = data
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                    
            result = self.cz_channel.calculate(df)
            return result.cer
            
        return None
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        if data is not None:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
                df = data
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                    
            result = self.cz_channel.calculate(df)
            return result.dynamic_multiplier
            
        return None
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATRの値
        """
        if data is not None:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
                df = data
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                    
            result = self.cz_channel.calculate(df)
            return result.catr
            
        return None
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.cz_channel.reset() if hasattr(self.cz_channel, 'reset') else None
        self._signals_cache = {} 