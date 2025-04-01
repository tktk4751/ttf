#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from strategies.base.signal_generator import BaseSignalGenerator
from signals.implementations.divergence.z_macd_divergence import ZMACDDivergenceSignal
from signals.implementations.divergence.z_macd_hidden_divergence import ZMACDHiddenDivergenceSignal
from signals.implementations.z_reversal_filter.filter import ZReversalFilterSignal
from signals.implementations.z_donchian.entry import ZDonchianBreakoutEntrySignal


@njit(parallel=True)
def calculate_entry_signals(
    divergence_signal: np.ndarray, 
    filter_signal: np.ndarray
) -> np.ndarray:
    """
    ZMACDダイバージェンスシグナルとZリバーサルフィルターシグナルを組み合わせて
    エントリーシグナルを計算します

    Args:
        divergence_signal: ZMACDダイバージェンスシグナル配列
        filter_signal: Zリバーサルフィルターシグナル配列

    Returns:
        np.ndarray: エントリーシグナル配列 (1=ロング, -1=ショート, 0=シグナルなし)
    """
    n = len(divergence_signal)
    entry_signals = np.zeros(n, dtype=np.int8)

    for i in prange(n):
        # ロングエントリー：両方のシグナルが1の場合
        if divergence_signal[i] == 1 and filter_signal[i] == 1:
            entry_signals[i] = 1
        
        # ショートエントリー：両方のシグナルが-1の場合
        elif divergence_signal[i] == -1 and filter_signal[i] == -1:
            entry_signals[i] = -1
    
    return entry_signals


class ZDivergenceSignalGenerator(BaseSignalGenerator):
    """
    Zダイバージェンス戦略のシグナル生成器
    
    特徴:
    - ZMACDダイバージェンスシグナルによるトレンド転換タイミングの検出
    - Zリバーサルフィルターによる市場状態の確認
    - Zドンチャンブレイクアウトによるエグジットポイントの判定
    - Numbaによる最適化で高速処理
    
    エントリーシグナル条件:
    - ロング: ZMACDダイバージェンスが1(強気ダイバージェンス) かつ Zリバーサルフィルターが1(ロングリバーサル)
    - ショート: ZMACDダイバージェンスが-1(弱気ダイバージェンス) かつ Zリバーサルフィルターが-1(ショートリバーサル)
    
    エグジットシグナル条件:
    - ロング: Zドンチャンブレイクアウトが-1(ショートブレイクアウト)
    - ショート: Zドンチャンブレイクアウトが1(ロングブレイクアウト)
    """
    
    def __init__(
        self,
        # ZMACDダイバージェンス用パラメータ
        er_period: int = 21,
        fast_max_dc_max_output: int = 21,
        fast_max_dc_min_output: int = 5,
        slow_max_dc_max_output: int = 55,
        slow_max_dc_min_output: int = 13,
        signal_max_dc_max_output: int = 21,
        signal_max_dc_min_output: int = 5,
        max_slow_period: int = 34,
        min_slow_period: int = 13,
        max_fast_period: int = 8,
        min_fast_period: int = 2,
        div_lookback: int = 30,
        
        # Zリバーサルフィルター用パラメータ
        # Zロングリバーサル用パラメータ
        long_max_rms_window: int = 13,
        long_min_rms_window: int = 5,
        long_max_threshold: float = 0.9,
        long_min_threshold: float = 0.75,
        
        # Zショートリバーサル用パラメータ
        short_max_rms_window: int = 13,
        short_min_rms_window: int = 5,
        short_max_threshold: float = 0.25,
        short_min_threshold: float = 0.1,
        
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        
        # 組み合わせパラメータ
        zadx_weight: float = 0.4,
        zrsx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        
        # ZADX用パラメータ
        zadx_max_dc_cycle_part: float = 0.5,
        zadx_max_dc_max_cycle: int = 34,
        zadx_max_dc_min_cycle: int = 5,
        zadx_max_dc_max_output: int = 21,
        zadx_max_dc_min_output: int = 8,
        zadx_min_dc_cycle_part: float = 0.25,
        zadx_min_dc_max_cycle: int = 21,
        zadx_min_dc_min_cycle: int = 3,
        zadx_min_dc_max_output: int = 13,
        zadx_min_dc_min_output: int = 3,
        zadx_er_period: int = 21,
        
        # ZRSX用パラメータ
        zrsx_max_dc_cycle_part: float = 0.5,
        zrsx_max_dc_max_cycle: int = 55,
        zrsx_max_dc_min_cycle: int = 5,
        zrsx_max_dc_max_output: int = 21,
        zrsx_max_dc_min_output: int = 10,
        zrsx_min_dc_cycle_part: float = 0.25,
        zrsx_min_dc_max_cycle: int = 34,
        zrsx_min_dc_min_cycle: int = 3,
        zrsx_min_dc_max_output: int = 10,
        zrsx_min_dc_min_output: int = 5,
        zrsx_er_period: int = 10,
        
        # Zドンチャン用パラメータ
        # 最大期間用パラメータ
        donchian_max_dc_cycle_part: float = 0.5,
        donchian_max_dc_max_cycle: int = 144,
        donchian_max_dc_min_cycle: int = 13,
        donchian_max_dc_max_output: int = 89,
        donchian_max_dc_min_output: int = 21,
        
        # 最小期間用パラメータ
        donchian_min_dc_cycle_part: float = 0.25,
        donchian_min_dc_max_cycle: int = 55,
        donchian_min_dc_min_cycle: int = 5,
        donchian_min_dc_max_output: int = 21,
        donchian_min_dc_min_output: int = 8,
        
        # ブレイクアウトパラメータ
        band_lookback: int = 1,
        
        # 共通パラメータ
        smoother_type: str = 'alma',  # 'alma'または'hyper'
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            多数のパラメータ - 各コンポーネントごとに必要なパラメータを受け取ります
        """
        super().__init__("ZDivergenceSignalGenerator")
        
        # ZMACDダイバージェンスシグナルの初期化
        self.macd_divergence = ZMACDDivergenceSignal(
            er_period=er_period,
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            signal_max_dc_max_output=signal_max_dc_max_output,
            signal_max_dc_min_output=signal_max_dc_min_output,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            lookback=div_lookback
        )
        
        # ZリバーサルフィルタシグナルのZ初期化
        self.reversal_filter = ZReversalFilterSignal(
            long_max_rms_window=long_max_rms_window,
            long_min_rms_window=long_min_rms_window,
            long_max_threshold=long_max_threshold,
            long_min_threshold=long_min_threshold,
            short_max_rms_window=short_max_rms_window,
            short_min_rms_window=short_min_rms_window,
            short_max_threshold=short_max_threshold,
            short_min_threshold=short_min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            zadx_weight=zadx_weight,
            zrsx_weight=zrsx_weight,
            combination_method=combination_method,
            zadx_max_dc_cycle_part=zadx_max_dc_cycle_part,
            zadx_max_dc_max_cycle=zadx_max_dc_max_cycle,
            zadx_max_dc_min_cycle=zadx_max_dc_min_cycle,
            zadx_max_dc_max_output=zadx_max_dc_max_output,
            zadx_max_dc_min_output=zadx_max_dc_min_output,
            zadx_min_dc_cycle_part=zadx_min_dc_cycle_part,
            zadx_min_dc_max_cycle=zadx_min_dc_max_cycle,
            zadx_min_dc_min_cycle=zadx_min_dc_min_cycle,
            zadx_min_dc_max_output=zadx_min_dc_max_output,
            zadx_min_dc_min_output=zadx_min_dc_min_output,
            zadx_er_period=zadx_er_period,
            zrsx_max_dc_cycle_part=zrsx_max_dc_cycle_part,
            zrsx_max_dc_max_cycle=zrsx_max_dc_max_cycle,
            zrsx_max_dc_min_cycle=zrsx_max_dc_min_cycle,
            zrsx_max_dc_max_output=zrsx_max_dc_max_output,
            zrsx_max_dc_min_output=zrsx_max_dc_min_output,
            zrsx_min_dc_cycle_part=zrsx_min_dc_cycle_part,
            zrsx_min_dc_max_cycle=zrsx_min_dc_max_cycle,
            zrsx_min_dc_min_cycle=zrsx_min_dc_min_cycle,
            zrsx_min_dc_max_output=zrsx_min_dc_max_output,
            zrsx_min_dc_min_output=zrsx_min_dc_min_output,
            zrsx_er_period=zrsx_er_period,
            smoother_type=smoother_type
        )
        
        # Zドンチャンブレイクアウトシグナルの初期化
        self.donchian_breakout = ZDonchianBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_dc_cycle_part=donchian_max_dc_cycle_part,
            max_dc_max_cycle=donchian_max_dc_max_cycle,
            max_dc_min_cycle=donchian_max_dc_min_cycle,
            max_dc_max_output=donchian_max_dc_max_output,
            max_dc_min_output=donchian_max_dc_min_output,
            min_dc_cycle_part=donchian_min_dc_cycle_part,
            min_dc_max_cycle=donchian_min_dc_max_cycle,
            min_dc_min_cycle=donchian_min_dc_min_cycle,
            min_dc_max_output=donchian_min_dc_max_output,
            min_dc_min_output=donchian_min_dc_min_output,
            lookback=band_lookback,
            src_type=src_type
        )
        
        # 結果を保存するための変数
        self._divergence_signals = None
        self._reversal_filter_signals = None
        self._donchian_signals = None
        self._entry_signals = None
        
        # データハッシュ用
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
            
        # データの形状に基づくハッシュの生成
        data_shape_str = f"{data_array.shape}_{data_array.dtype}"
        
        return hashlib.md5(data_shape_str.encode()).hexdigest()
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        すべてのシグナルを計算する

        Args:
            data: 価格データ
        """
        try:
            # データのハッシュ値を計算
            data_hash = self._get_data_hash(data)
            
            # 同じデータでキャッシュが存在する場合、再計算しない
            if self._data_hash == data_hash and self._entry_signals is not None:
                return
            
            # ハッシュを更新
            self._data_hash = data_hash
            
            # 各シグナルの計算
            self._divergence_signals = self.macd_divergence.generate(data)
            self._reversal_filter_signals = self.reversal_filter.generate(data)
            self._donchian_signals = self.donchian_breakout.generate(data)
            
            # ダイバージェンスシグナルとリバーサルフィルタを組み合わせてエントリーシグナルを生成
            self._entry_signals = calculate_entry_signals(
                self._divergence_signals,
                self._reversal_filter_signals
            )
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"シグナル計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の配列をセット
            if isinstance(data, pd.DataFrame):
                length = len(data)
            else:
                length = data.shape[0]
                
            self._divergence_signals = np.zeros(length)
            self._reversal_filter_signals = np.zeros(length)
            self._donchian_signals = np.zeros(length)
            self._entry_signals = np.zeros(length)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得

        Args:
            data: 価格データ

        Returns:
            np.ndarray: エントリーシグナル配列
        """
        self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを取得

        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）

        Returns:
            bool: エグジットすべきかどうか
        """
        self.calculate_signals(data)
        
        # インデックスが範囲外の場合はFalseを返す
        if index < 0:
            index = len(self._donchian_signals) - 1
        
        if index >= len(self._donchian_signals):
            return False
        
        # ロングポジションの場合、ドンチャンブレイクアウトが-1（ショートブレイクアウト）ならエグジット
        if position == 1 and self._donchian_signals[index] == -1:
            return True
        
        # ショートポジションの場合、ドンチャンブレイクアウトが1（ロングブレイクアウト）ならエグジット
        if position == -1 and self._donchian_signals[index] == 1:
            return True
        
        return False
    
    def get_divergence_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZMACDダイバージェンスシグナルを取得

        Args:
            data: 価格データ（キャッシュがあればNoneで良い）

        Returns:
            np.ndarray: ZMACDダイバージェンスシグナル配列
        """
        if data is not None:
            self.calculate_signals(data)
        return self._divergence_signals if self._divergence_signals is not None else np.array([])
    
    def get_reversal_filter_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zリバーサルフィルタシグナルを取得

        Args:
            data: 価格データ（キャッシュがあればNoneで良い）

        Returns:
            np.ndarray: Zリバーサルフィルタシグナル配列
        """
        if data is not None:
            self.calculate_signals(data)
        return self._reversal_filter_signals if self._reversal_filter_signals is not None else np.array([])
    
    def get_donchian_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zドンチャンブレイクアウトシグナルを取得

        Args:
            data: 価格データ（キャッシュがあればNoneで良い）

        Returns:
            np.ndarray: Zドンチャンブレイクアウトシグナル配列
        """
        if data is not None:
            self.calculate_signals(data)
        return self._donchian_signals if self._donchian_signals is not None else np.array([])
    
    def get_donchian_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zドンチャンバンドを取得

        Args:
            data: 価格データ（キャッシュがあればNoneで良い）

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (上限バンド, 下限バンド, 中央線)のタプル
        """
        if data is not None:
            self.calculate_signals(data)
        return self.donchian_breakout.get_band_values()
    
    def get_macd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        ZMACD値を取得

        Args:
            data: 価格データ（キャッシュがあればNoneで良い）

        Returns:
            Dict[str, np.ndarray]: ZMACD値を含む辞書
        """
        if data is not None:
            self.calculate_signals(data)
        return self.macd_divergence.get_z_macd_values(data)
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（ER）を取得

        Args:
            data: 価格データ（キャッシュがあればNoneで良い）

        Returns:
            np.ndarray: 効率比の配列
        """
        if data is not None:
            self.calculate_signals(data)
        return self.donchian_breakout.get_efficiency_ratio()
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        self.macd_divergence = None  # メモリ節約のためNoneに設定
        self.reversal_filter = None
        self.donchian_breakout = None
        self._divergence_signals = None
        self._reversal_filter_signals = None
        self._donchian_signals = None
        self._entry_signals = None
        self._data_hash = None 