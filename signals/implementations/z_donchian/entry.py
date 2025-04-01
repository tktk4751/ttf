#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_donchian import ZDonchian
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


@njit(fastmath=True, parallel=True)
def generate_breakout_signals_numba(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    lookback: int
) -> np.ndarray:
    """
    Numbaによる高速なブレイクアウトシグナル生成

    Args:
        close: 終値の配列
        upper: 上限バンドの配列
        lower: 下限バンドの配列
        lookback: 遡る期間（デフォルト: 1）

    Returns:
        シグナル配列 (1: ロング, -1: ショート, 0: ニュートラル)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int64)
    
    # 最初の部分はシグナルなし（十分なデータがないため）
    min_idx = max(lookback, 1)
    
    for i in prange(min_idx, length):
        # 有効なデータがあるか確認
        if np.isnan(close[i]) or np.isnan(upper[i-lookback]) or np.isnan(lower[i-lookback]):
            signals[i] = 0
            continue
        
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-lookback]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-lookback]:
            signals[i] = -1
        # それ以外はシグナルなし
        else:
            signals[i] = 0
    
    return signals


class ZDonchianBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    Zドンチャンチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - 効率比（ER）に基づいて動的に期間が調整されるZドンチャンチャネルを使用
    - 現在の終値がZドンチャンのアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値がZドンチャンのロワーバンドを下回った場合: ショートエントリー (-1)
    - Numbaによる最適化で高速処理
    - トレンドの強さに応じて適応的なブレイクアウト判定
    
    パラメータ:
    - Z_Donchianのパラメータ（最大/最小期間用ドミナントサイクル設定）
    - サイクル効率比（CER）の計算パラメータ
    - lookback: 何期間前のバンドと比較するか（デフォルト: 1）
    """
    
    def __init__(
        self,
        # ドミナントサイクル・効率比（CER）の基本パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # 最大期間用パラメータ
        max_dc_cycle_part: float = 0.5,
        max_dc_max_cycle: int = 144,
        max_dc_min_cycle: int = 13,
        max_dc_max_output: int = 89,
        max_dc_min_output: int = 21,
        
        # 最小期間用パラメータ
        min_dc_cycle_part: float = 0.25,
        min_dc_max_cycle: int = 55,
        min_dc_min_cycle: int = 5,
        min_dc_max_output: int = 21,
        min_dc_min_output: int = 8,
        
        # ブレイクアウトパラメータ
        lookback: int = 1,
        
        # ソースタイプ
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: ドミナントサイクル検出アルゴリズム
            lp_period: 効率比計算用ローパスフィルター期間
            hp_period: 効率比計算用ハイパスフィルター期間
            cycle_part: ドミナントサイクルの一部として使用する割合
            
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算用のサイクル部分
            max_dc_max_cycle: 最大期間用ドミナントサイクル計算用の最大サイクル期間
            max_dc_min_cycle: 最大期間用ドミナントサイクル計算用の最小サイクル期間
            max_dc_max_output: 最大期間用ドミナントサイクル計算用の最大出力値
            max_dc_min_output: 最大期間用ドミナントサイクル計算用の最小出力値
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算用のサイクル部分
            min_dc_max_cycle: 最小期間用ドミナントサイクル計算用の最大サイクル期間
            min_dc_min_cycle: 最小期間用ドミナントサイクル計算用の最小サイクル期間
            min_dc_max_output: 最小期間用ドミナントサイクル計算用の最大出力値
            min_dc_min_output: 最小期間用ドミナントサイクル計算用の最小出力値
            
            lookback: 何期間前のバンドと比較するか
            src_type: 価格計算の元となる価格タイプ
        """
        super().__init__(f"ZDonchianBreakout({max_dc_max_output}-{min_dc_min_output})")
        
        # パラメータの保存
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        self.max_dc_cycle_part = max_dc_cycle_part
        self.max_dc_max_cycle = max_dc_max_cycle
        self.max_dc_min_cycle = max_dc_min_cycle
        self.max_dc_max_output = max_dc_max_output
        self.max_dc_min_output = max_dc_min_output
        
        self.min_dc_cycle_part = min_dc_cycle_part
        self.min_dc_max_cycle = min_dc_max_cycle
        self.min_dc_min_cycle = min_dc_min_cycle
        self.min_dc_max_output = min_dc_max_output
        self.min_dc_min_output = min_dc_min_output
        
        self.lookback = lookback
        self.src_type = src_type
        
        # インジケーターの初期化
        self.z_donchian = ZDonchian(
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=max_dc_max_output,
            max_dc_min_output=max_dc_min_output,
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=min_dc_max_output,
            min_dc_min_output=min_dc_min_output
        )
        
        # サイクル効率比（CER）の初期化
        self.cer = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            src_type=src_type
        )
        
        # 結果を保存する属性
        self._signals = None
        self._upper = None
        self._lower = None
        self._middle = None
        self._cer_values = None
        self._dynamic_period = None
        
        # キャッシュ用の属性
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算する

        Args:
            data: データ

        Returns:
            str: ハッシュ値
        """
        import hashlib
        
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合はnumpy配列に変換
            data_array = data.values
        else:
            data_array = data
            
        # データの形状とパラメータを含めたハッシュの生成
        param_str = (
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.max_dc_cycle_part}_{self.max_dc_max_cycle}_{self.max_dc_min_cycle}_"
            f"{self.max_dc_max_output}_{self.max_dc_min_output}_"
            f"{self.min_dc_cycle_part}_{self.min_dc_max_cycle}_{self.min_dc_min_cycle}_"
            f"{self.min_dc_max_output}_{self.min_dc_min_output}_{self.lookback}"
        )
        data_shape_str = f"{data_array.shape}_{data_array.dtype}"
        hash_str = f"{param_str}_{data_shape_str}"
        
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zドンチャンブレイクアウトシグナルを生成

        Args:
            data: 価格データ

        Returns:
            np.ndarray: シグナル配列 (1: ロング, -1: ショート, 0: ニュートラル)
        """
        try:
            # データのハッシュ値を計算
            data_hash = self._get_data_hash(data)
            
            # 同じデータでキャッシュが存在する場合、キャッシュを返す
            if self._data_hash == data_hash and self._signals is not None:
                return self._signals
            
            # ハッシュを更新
            self._data_hash = data_hash
            
            # 効率比（CER）の計算
            cer_values = self.cer.calculate(data)
            
            # Zドンチャンチャネルの計算
            self.z_donchian.calculate(data, external_er=cer_values)
            upper, lower, middle = self.z_donchian.get_bands()
            
            # 終値の取得
            if isinstance(data, pd.DataFrame):
                close = data['close'].values
            else:
                close = data[:, 3]  # OHLC形式を想定
            
            # ブレイクアウトシグナルの生成（Numba並列処理）
            signals = generate_breakout_signals_numba(
                close,
                upper,
                lower,
                self.lookback
            )
            
            # 結果を保存
            self._signals = signals
            self._upper = upper
            self._lower = lower
            self._middle = middle
            self._cer_values = cer_values
            self._dynamic_period = self.z_donchian.get_dynamic_period()
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"Zドンチャンブレイクアウトシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はゼロシグナルを返す
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(data.shape[0])
    
    def get_band_values(self) -> tuple:
        """
        バンド値を取得
        
        Returns:
            tuple: (上限バンド, 下限バンド, 中央線)のタプル
        """
        if self._upper is None or self._lower is None or self._middle is None:
            return np.array([]), np.array([]), np.array([])
        return self._upper, self._lower, self._middle
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比を取得
        
        Returns:
            np.ndarray: 効率比（CER）の値
        """
        return self._cer_values if self._cer_values is not None else np.array([])
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的な期間を取得
        
        Returns:
            np.ndarray: 動的な期間の値
        """
        return self._dynamic_period if self._dynamic_period is not None else np.array([])
    
    def get_signals(self) -> np.ndarray:
        """
        シグナル配列を取得
        
        Returns:
            np.ndarray: シグナル配列
        """
        return self._signals if self._signals is not None else np.array([])
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        self.z_donchian.reset()
        self.cer.reset()
        self._signals = None
        self._upper = None
        self._lower = None
        self._middle = None
        self._cer_values = None
        self._dynamic_period = None
        self._data_hash = None 