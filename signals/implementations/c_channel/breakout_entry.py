#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
import hashlib

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.c_channel import CChannel


@njit(fastmath=True, parallel=True)
def calculate_breakout_signals(close: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray, lookback: int = 1) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（Numba高速化版）
    
    Args:
        close: 終値の配列
        upper_band: 上限バンドの配列
        lower_band: 下限バンドの配列
        lookback: バンド参照期間（過去のバンドと比較、デフォルト: 1）
        
    Returns:
        np.ndarray: シグナル値の配列
            1: 上向きブレイクアウト（終値が上限バンドを超えた）
            -1: 下向きブレイクアウト（終値が下限バンドを下回った）
            0: シグナルなし
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    
    # シグナル計算開始位置（十分なデータがある場所から）
    start_idx = max(lookback, 1)
    
    # 各バーでの判定を並列処理
    for i in prange(start_idx, n):
        # NaNチェック
        if np.isnan(close[i]) or np.isnan(upper_band[i-lookback]) or np.isnan(lower_band[i-lookback]):
            signals[i] = 0
            continue
            
        # ブレイクアウト条件
        if close[i] > upper_band[i-lookback]:  # 上限ブレイク（買い）
            signals[i] = 1
        elif close[i] < lower_band[i-lookback]:  # 下限ブレイク（売り）
            signals[i] = -1
        else:
            signals[i] = 0
            
    return signals


class CCBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    Cチャネルのブレイクアウトエントリーシグナル
    
    特徴:
    - Cチャネル（CMA + CATR）を使用したブレイクアウト戦略
    - サイクル効率比（CER）に基づく動的ATR乗数でバンド幅を調整
    - Numbaによる高速化処理
    
    シグナル条件:
    - 買い: 終値が上限バンドを上抜けた場合
    - 売り: 終値が下限バンドを下抜けた場合
    """
    
    def __init__(
        self,
        # Cチャネル共通パラメータ
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
        
        # CMA用パラメータ
        cma_detector_type: str = 'hody_e',
        cma_cycle_part: float = 0.5,
        cma_lp_period: int = 5,
        cma_hp_period: int = 55,
        cma_max_cycle: int = 144,
        cma_min_cycle: int = 5,
        cma_max_output: int = 62,
        cma_min_output: int = 13,
        cma_fast_period: int = 2,
        cma_slow_period: int = 30,
        cma_src_type: str = 'hlc3',
        
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
        コンストラクタ
        
        Args:
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積（デフォルト）
            cer_detector_type: CER用の検出器タイプ（Noneの場合はdetector_typeと同じ）
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分の倍率
            smoother_type: 平滑化アルゴリズム ('alma'または'hyper')
            src_type: 価格ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            band_lookback: バンド参照期間（過去のバンドと比較する際のオフセット）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            
            # CMAパラメータ
            cma_detector_type: CMA用検出器タイプ
            cma_cycle_part: CMA用サイクル部分の倍率
            cma_lp_period: CMA用ローパスフィルター期間
            cma_hp_period: CMA用ハイパスフィルター期間
            cma_max_cycle: CMA用最大サイクル期間
            cma_min_cycle: CMA用最小サイクル期間
            cma_max_output: CMA用最大出力値
            cma_min_output: CMA用最小出力値
            cma_fast_period: CMA用速い移動平均の期間
            cma_slow_period: CMA用遅い移動平均の期間
            cma_src_type: CMA用ソースタイプ
            
            # CATRパラメータ
            catr_detector_type: CATR用検出器タイプ
            catr_cycle_part: CATR用サイクル部分の倍率
            catr_lp_period: CATR用ローパスフィルター期間
            catr_hp_period: CATR用ハイパスフィルター期間
            catr_max_cycle: CATR用最大サイクル期間
            catr_min_cycle: CATR用最小サイクル期間
            catr_max_output: CATR用最大出力値
            catr_min_output: CATR用最小出力値
            catr_smoother_type: CATR用平滑化タイプ
        """
        super().__init__("CCBreakoutEntrySignal")
        
        # 基本パラメータの設定
        self.band_lookback = max(1, band_lookback)  # 最小値は1
        
        # CER検出器タイプの設定（指定されていない場合はdetector_typeを使用）
        if cer_detector_type is None:
            cer_detector_type = detector_type
        
        # パラメータの保存
        self._params = {
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # CMA用パラメータ
            'cma_detector_type': cma_detector_type,
            'cma_cycle_part': cma_cycle_part,
            'cma_lp_period': cma_lp_period,
            'cma_hp_period': cma_hp_period,
            'cma_max_cycle': cma_max_cycle,
            'cma_min_cycle': cma_min_cycle,
            'cma_max_output': cma_max_output,
            'cma_min_output': cma_min_output,
            'cma_fast_period': cma_fast_period,
            'cma_slow_period': cma_slow_period,
            'cma_src_type': cma_src_type,
            
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
        
        # Cチャネルの初期化
        self.c_channel = CChannel(
            # 基本パラメータ
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
            
            # CMA用パラメータ
            cma_detector_type=cma_detector_type,
            cma_cycle_part=cma_cycle_part,
            cma_lp_period=cma_lp_period,
            cma_hp_period=cma_hp_period,
            cma_max_cycle=cma_max_cycle,
            cma_min_cycle=cma_min_cycle,
            cma_max_output=cma_max_output,
            cma_min_output=cma_min_output,
            cma_fast_period=cma_fast_period,
            cma_slow_period=cma_slow_period,
            cma_src_type=cma_src_type,
            
            # CATR用パラメータ
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
        
        # キャッシュの初期化
        self._signals = None
        self._data_hash = None
    
    def _get_data_hash(self, ohlcv_data):
        """データのハッシュ値を計算（キャッシュ用）"""
        if isinstance(ohlcv_data, pd.DataFrame):
            # pandas DataFrameの場合、基本的なOHLCVカラムをハッシュ化
            try:
                # 日時インデックスがある場合はインデックスハッシュも含める
                if isinstance(ohlcv_data.index, pd.DatetimeIndex):
                    index_hash = hash(tuple(ohlcv_data.index))
                else:
                    index_hash = 0
                
                # 必要なカラムのハッシュを計算
                data_hash = hash(tuple(ohlcv_data['close'].values.tobytes()))
                
                # パラメータハッシュと組み合わせる
                params_hash = hash((self._params.get('band_lookback', 1),))
                
                return f"{index_hash}_{data_hash}_{params_hash}"
            except Exception:
                # ハッシュ計算に失敗した場合はデータのIDを使用
                return id(ohlcv_data)
        else:
            # NumPy配列の場合、バイトデータをハッシュ化
            try:
                return hash((ohlcv_data.tobytes(), self._params.get('band_lookback', 1)))
            except Exception:
                return id(ohlcv_data)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ（DataFrameまたはnumpy配列）
            
        Returns:
            np.ndarray: シグナル値の配列
                1: 上向きブレイクアウト（終値が上限バンドを超えた）
                -1: 下向きブレイクアウト（終値が下限バンドを下回った）
                0: シグナルなし
        """
        try:
            # データのハッシュ値を計算
            if isinstance(data, pd.DataFrame):
                data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
            else:
                data_hash = hashlib.md5(data.tobytes()).hexdigest()
            
            # キャッシュが有効な場合は、計算済みの結果を返す
            if self._data_hash is not None and self._data_hash == data_hash and self._signals is not None:
                return self._signals
            
            # データフレームの作成（必要な列のみ）
            if isinstance(data, pd.DataFrame):
                df = data[['open', 'high', 'low', 'close']].copy()
            else:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            
            # Cチャネルの計算
            self.c_channel.calculate(df)
            
            # バンド値の取得
            middle, upper, lower = self.c_channel.get_bands()
            
            # ブレイクアウトシグナルの計算
            close = df['close'].values
            band_lookback = self._params.get('band_lookback', 1)
            band_lookback = max(1, band_lookback)  # 最低1を保証
            signals = calculate_breakout_signals(close, upper, lower, band_lookback)
            
            # キャッシュの更新
            self._data_hash = data_hash
            self._signals = signals
            
            return signals
        except Exception as e:
            import traceback
            print(f"シグナル生成中にエラー: {str(e)}\n{traceback.format_exc()}")
            if isinstance(data, (pd.DataFrame, np.ndarray)):
                return np.zeros(len(data), dtype=np.int8)
            return np.array([], dtype=np.int8)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.generate(data)
        
        return self.c_channel.get_bands()
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            if data is not None:
                # データが指定された場合は再計算
                self.generate(data)
            
            return self.c_channel.get_cycle_er()
        except Exception as e:
            import traceback
            print(f"効率比取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        try:
            if data is not None:
                # データが指定された場合は再計算
                self.generate(data)
            
            return self.c_channel.get_dynamic_multiplier()
        except Exception as e:
            import traceback
            print(f"動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR値
        """
        try:
            if data is not None:
                # データが指定された場合は再計算
                self.generate(data)
            
            return self.c_channel.get_c_atr()
        except Exception as e:
            import traceback
            print(f"CATR取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def reset(self) -> None:
        """
        シグナル生成器の状態をリセットする
        """
        super().reset()
        self._signals = None
        self._data_hash = None
        self.c_channel.reset() if hasattr(self.c_channel, 'reset') else None 