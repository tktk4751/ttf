#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
# 使用するチャネルインジケーターをインポート
from indicators.cc_channel import CC_Channel 


# calculate_breakout_signals 関数は ZChannelBreakoutEntrySignal から流用可能
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
        シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
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


class CCChannelBreakoutEntrySignal(BaseSignal, IEntrySignal): # クラス名を変更
    """
    CCチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - 中心線にCMA、バンド幅にCATRを使用
    - サイクルボラティリティ効率比 (CVER) に基づく動的な適応性
    - 使用するインジケーターのパラメータは外部から設定可能
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__( # 引数を band_lookback のみに変更 -> CC_Channel のパラメータも受け取るように変更
        self,
        band_lookback: int = 1, 
        # CC_Channel のパラメータを追加
        multiplier_method: str = 'adaptive', # multiplier_method を追加
        new_method_er_source: str = 'cver', # new_method_er_source を追加
        cc_max_max_multiplier: float = 8.0,
        cc_min_max_multiplier: float = 3.0,
        cc_max_min_multiplier: float = 1.5,
        cc_min_min_multiplier: float = 0.5,
        cma_detector_type: str = 'hody_e', 
        cma_cycle_part: float = 0.618, 
        cma_lp_period: int = 5, 
        cma_hp_period: int = 89, 
        cma_max_cycle: int = 55, 
        cma_min_cycle: int = 5, 
        cma_max_output: int = 34, 
        cma_min_output: int = 8, 
        cma_fast_period: int = 2, 
        cma_slow_period: int = 30, 
        cma_src_type: str = 'hlc3',
        catr_detector_type: str = 'hody', 
        catr_cycle_part: float = 0.5, 
        catr_lp_period: int = 5, 
        catr_hp_period: int = 55, 
        catr_max_cycle: int = 55, 
        catr_min_cycle: int = 5, 
        catr_max_output: int = 34, 
        catr_min_output: int = 5, 
        catr_smoother_type: str = 'alma',
        cver_detector_type: str = 'hody', 
        cver_lp_period: int = 5, 
        cver_hp_period: int = 144, 
        cver_cycle_part: float = 0.5, 
        cver_max_cycle: int = 144, 
        cver_min_cycle: int = 5, 
        cver_max_output: int = 89, 
        cver_min_output: int = 5, 
        cver_src_type: str = 'hlc3',
        cer_detector_type: str = 'hody', 
        cer_lp_period: int = 5,
        cer_hp_period: int = 144,
        cer_cycle_part: float = 0.5,
        cer_max_cycle: int = 144,
        cer_min_cycle: int = 5,
        cer_max_output: int = 89,
        cer_min_output: int = 5,
        cer_src_type: str = 'hlc3'
    ):
        """
        初期化
        
        Args:
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            **kwargs: CC_Channel の初期化パラメータ
        """
        super().__init__(
            f"CCChannelBreakoutEntrySignal(lookback={band_lookback}, method={multiplier_method}, er_src={new_method_er_source if multiplier_method == 'new' else 'n/a'})" # 識別子を更新
        )
        
        # パラメータの保存 (band_lookback と CC_Channel のパラメータ)
        self._params = {
            'band_lookback': band_lookback,
            'multiplier_method': multiplier_method, # multiplier_method を追加
            'new_method_er_source': new_method_er_source, # new_method_er_source を追加
            'cc_max_max_multiplier': cc_max_max_multiplier,
            'cc_min_max_multiplier': cc_min_max_multiplier,
            'cc_max_min_multiplier': cc_max_min_multiplier,
            'cc_min_min_multiplier': cc_min_min_multiplier,
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
            'catr_detector_type': catr_detector_type, 
            'catr_cycle_part': catr_cycle_part, 
            'catr_lp_period': catr_lp_period, 
            'catr_hp_period': catr_hp_period, 
            'catr_max_cycle': catr_max_cycle, 
            'catr_min_cycle': catr_min_cycle, 
            'catr_max_output': catr_max_output, 
            'catr_min_output': catr_min_output, 
            'catr_smoother_type': catr_smoother_type,
            'cver_detector_type': cver_detector_type, 
            'cver_lp_period': cver_lp_period, 
            'cver_hp_period': cver_hp_period, 
            'cver_cycle_part': cver_cycle_part, 
            'cver_max_cycle': cver_max_cycle, 
            'cver_min_cycle': cver_min_cycle, 
            'cver_max_output': cver_max_output, 
            'cver_min_output': cver_min_output, 
            'cver_src_type': cver_src_type,
            'cer_detector_type': cer_detector_type, 
            'cer_lp_period': cer_lp_period,
            'cer_hp_period': cer_hp_period,
            'cer_cycle_part': cer_cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'cer_src_type': cer_src_type
        }
            
        # CCチャネルの初期化 (パラメータを渡す)
        self.cc_channel = CC_Channel(
            multiplier_method=multiplier_method, # multiplier_method を渡す
            new_method_er_source=new_method_er_source, # new_method_er_source を渡す
            cc_max_max_multiplier=cc_max_max_multiplier,
            cc_min_max_multiplier=cc_min_max_multiplier,
            cc_max_min_multiplier=cc_max_min_multiplier,
            cc_min_min_multiplier=cc_min_min_multiplier,
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
            catr_detector_type=catr_detector_type, 
            catr_cycle_part=catr_cycle_part, 
            catr_lp_period=catr_lp_period, 
            catr_hp_period=catr_hp_period, 
            catr_max_cycle=catr_max_cycle, 
            catr_min_cycle=catr_min_cycle, 
            catr_max_output=catr_max_output, 
            catr_min_output=catr_min_output, 
            catr_smoother_type=catr_smoother_type,
            cver_detector_type=cver_detector_type, 
            cver_lp_period=cver_lp_period, 
            cver_hp_period=cver_hp_period, 
            cver_cycle_part=cver_cycle_part, 
            cver_max_cycle=cver_max_cycle, 
            cver_min_cycle=cver_min_cycle, 
            cver_max_output=cver_max_output, 
            cver_min_output=cver_min_output, 
            cver_src_type=cver_src_type,
            cer_detector_type=cer_detector_type, 
            cer_lp_period=cer_lp_period,
            cer_hp_period=cer_hp_period,
            cer_cycle_part=cer_cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            cer_src_type=cer_src_type
        )
        
        # 参照期間の設定
        self.band_lookback = band_lookback
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data): # ZChannel版から流用、パラメータ部分を修正
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムを特定 (CC_Channelが内部で使う可能性のあるもの)
            cols = ['open', 'high', 'low', 'close', 'volume']
            relevant_cols = [col for col in cols if col in ohlcv_data.columns]
            if not relevant_cols:
                raise ValueError("DataFrameに必要なカラム (open, high, low, closeなど) が見つかりません")
            # NumPy配列に変換してハッシュ計算（高速化）
            ohlcv_array = ohlcv_data[relevant_cols].values
        else:
            # NumPy配列でない場合はエラーを発生させる前に型チェック
            if not isinstance(ohlcv_data, np.ndarray):
                 raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
            ohlcv_array = ohlcv_data
            
        # NumPy配列かどうかの最終チェック
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと全パラメータのハッシュを組み合わせる (順序固定のためソート)
        param_hash = hash(tuple(sorted(self._params.items())))
        return hash((ohlcv_array.tobytes(), param_hash))
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # CCチャネルの計算 (内部でキャッシュが効くはず)
            # calculate は中心線 (CMA) を返す
            center_line = self.cc_channel.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if center_line is None or len(center_line) == 0:
                 # エラーは cc_channel 内部でログされる想定
                 self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                 return self._signals_cache[data_hash]
            
            # 終値の取得 (calculate_breakout_signals で必要)
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameに 'close' カラムが必要です")
                close = data['close'].values
            elif data.ndim == 2 and data.shape[1] >= 4:
                 close = data[:, 3] # OHLC想定
            else:
                 # より明確なエラーメッセージ
                 raise ValueError(f"データ形式が不正です (shape: {data.shape if hasattr(data, 'shape') else 'N/A'}). DataFrame (closeカラム含む) または Numpy配列 (OHLC形式) である必要があります。")
            
            # バンドの取得 (get_bands は内部で calculate を呼ぶ可能性があるが、
            # cc_channel のキャッシュが効くはず)
            _, upper, lower = self.cc_channel.get_bands(data) # 念のため data を渡す
            
            # バンド取得失敗チェック
            if upper is None or lower is None or len(upper) == 0 or len(lower) == 0:
                self.logger.warning("CCチャネルのバンド取得に失敗しました。ゼロシグナルを返します。")
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # ブレイクアウトシグナルの計算（高速化版）
            signals = calculate_breakout_signals(
                close,
                upper,
                lower,
                self.band_lookback
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CCChannelBreakoutEntrySignal計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時はキャッシュせずにゼロシグナルを返す
            return np.zeros(len(data), dtype=np.int8)
    
    # --- ゲッターメソッド (CC_Channel のゲッターをラップ) --- 
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CCチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線(CMA), 上限バンド, 下限バンド)のタプル
        """
        # CC_Channel の get_bands を直接呼び出す
        return self.cc_channel.get_bands(data)
    
    def get_cycle_volatility_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルボラティリティ効率比 (CVER) の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CVERの値
        """
        return self.cc_channel.get_cycle_volatility_er(data)
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        return self.cc_channel.get_dynamic_multiplier(data)
    
    def get_catr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値 (絶対値/金額ベース) を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR絶対値
        """
        return self.cc_channel.get_catr(data)
        
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最大ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最大ATR乗数の値
        """
        return self.cc_channel.get_dynamic_max_multiplier(data)
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最小ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最小ATR乗数の値
        """
        return self.cc_channel.get_dynamic_min_multiplier(data)
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.cc_channel.reset() if hasattr(self.cc_channel, 'reset') else None
        self._signals_cache = {} 