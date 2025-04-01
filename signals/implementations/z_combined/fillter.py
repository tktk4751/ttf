#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_bollinger_bands import ZBollingerBands
from indicators.z_channel import ZChannel


@njit(fastmath=True, parallel=True)
def calculate_comparison_signals(
    bb_upper: np.ndarray, 
    bb_lower: np.ndarray, 
    channel_upper: np.ndarray, 
    channel_lower: np.ndarray
) -> np.ndarray:
    """
    ZBBとZChannelの位置関係を比較してシグナルを計算する（高速化版）
    
    Args:
        bb_upper: ZBBの上限バンドの配列
        bb_lower: ZBBの下限バンドの配列
        channel_upper: ZChannelの上限バンドの配列
        channel_lower: ZChannelの下限バンドの配列
    
    Returns:
        シグナルの配列 (1: BBがチャネルの外側, -1: BBがチャネルの内側, 0: 判定不能)
    """
    length = len(bb_upper)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # 全ての値が有効かチェック
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(channel_upper[i]) or np.isnan(channel_lower[i])):
            signals[i] = 0
            continue
            
        # BBがチャネルの外側にある場合（BB幅 > チャネル幅）
        if bb_upper[i] > channel_upper[i] and bb_lower[i] < channel_lower[i]:
            signals[i] = 1
        # BBがチャネルの内側にある場合（BB幅 < チャネル幅）
        elif bb_upper[i] < channel_upper[i] and bb_lower[i] > channel_lower[i]:
            signals[i] = -1
        # その他の場合（一部重複など）は0
        else:
            signals[i] = 0
    
    return signals


class ZMomentumFillterSignal(BaseSignal, IEntrySignal):
    """
    ZBBとZChannelの位置関係を比較するシグナル
    
    特徴:
    - ZBBとZChannelのバンド幅の関係を分析
    - サイクル効率比（CER）に基づく両指標の動的な適応性を活用
    - Numba最適化による高速計算
    
    シグナル条件:
    - ZBBがZChannelの外側にある場合（BB幅 > チャネル幅）: 1
    - ZBBがZChannelの内側にある場合（BB幅 < チャネル幅）: -1
    - どちらとも判断できない場合（一部重複など）: 0
    
    トレード戦略応用例:
    - シグナル1の場合: ボラティリティが高く、レンジ相場の可能性
    - シグナル-1の場合: ボラティリティが低く、トレンド相場の可能性
    """
    
    def __init__(
        self,
        # 共通パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        src_type: str = 'hlc3',
        
        # ZBBのパラメータ
        zbb_max_multiplier: float = 2.5,
        zbb_min_multiplier: float = 1.0,
        zbb_max_std_dev_cycle_part: float = 0.5,
        zbb_max_std_dev_max_cycle: int = 144,
        zbb_max_std_dev_min_cycle: int = 10,
        zbb_max_std_dev_max_output: int = 89,
        zbb_max_std_dev_min_output: int = 13,
        zbb_min_std_dev_cycle_part: float = 0.25,
        zbb_min_std_dev_max_cycle: int = 55,
        zbb_min_std_dev_min_cycle: int = 5,
        zbb_min_std_dev_max_output: int = 21,
        zbb_min_std_dev_min_output: int = 5,
        
        # ZChannelのパラメータ
        zc_max_multiplier: float = 3.0,
        zc_min_multiplier: float = 1.5,
        zc_smoother_type: str = 'alma'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            src_type: ソースタイプ （デフォルト: 'hlc3'）
            
            zbb_max_multiplier: ZBBの標準偏差乗数の最大値（デフォルト: 2.5）
            zbb_min_multiplier: ZBBの標準偏差乗数の最小値（デフォルト: 1.0）
            zbb_max_std_dev_cycle_part: ZBBの最大標準偏差サイクル部分（デフォルト: 0.5）
            zbb_max_std_dev_max_cycle: ZBBの最大標準偏差最大サイクル（デフォルト: 144）
            zbb_max_std_dev_min_cycle: ZBBの最大標準偏差最小サイクル（デフォルト: 10）
            zbb_max_std_dev_max_output: ZBBの最大標準偏差最大出力（デフォルト: 89）
            zbb_max_std_dev_min_output: ZBBの最大標準偏差最小出力（デフォルト: 13）
            zbb_min_std_dev_cycle_part: ZBBの最小標準偏差サイクル部分（デフォルト: 0.25）
            zbb_min_std_dev_max_cycle: ZBBの最小標準偏差最大サイクル（デフォルト: 55）
            zbb_min_std_dev_min_cycle: ZBBの最小標準偏差最小サイクル（デフォルト: 5）
            zbb_min_std_dev_max_output: ZBBの最小標準偏差最大出力（デフォルト: 21）
            zbb_min_std_dev_min_output: ZBBの最小標準偏差最小出力（デフォルト: 5）
            
            zc_max_multiplier: ZChannelのATR乗数の最大値（デフォルト: 3.0）
            zc_min_multiplier: ZChannelのATR乗数の最小値（デフォルト: 1.5）
            zc_smoother_type: ZChannelの平滑化アルゴリズム（デフォルト: 'alma'）
        """
        params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'src_type': src_type,
            'zbb_max_multiplier': zbb_max_multiplier,
            'zbb_min_multiplier': zbb_min_multiplier,
            'zbb_max_std_dev_cycle_part': zbb_max_std_dev_cycle_part,
            'zbb_max_std_dev_max_cycle': zbb_max_std_dev_max_cycle,
            'zbb_max_std_dev_min_cycle': zbb_max_std_dev_min_cycle,
            'zbb_max_std_dev_max_output': zbb_max_std_dev_max_output,
            'zbb_max_std_dev_min_output': zbb_max_std_dev_min_output,
            'zbb_min_std_dev_cycle_part': zbb_min_std_dev_cycle_part,
            'zbb_min_std_dev_max_cycle': zbb_min_std_dev_max_cycle,
            'zbb_min_std_dev_min_cycle': zbb_min_std_dev_min_cycle,
            'zbb_min_std_dev_max_output': zbb_min_std_dev_max_output,
            'zbb_min_std_dev_min_output': zbb_min_std_dev_min_output,
            'zc_max_multiplier': zc_max_multiplier,
            'zc_min_multiplier': zc_min_multiplier,
            'zc_smoother_type': zc_smoother_type
        }
        super().__init__(
            f"ZBBChannelComparison({cycle_detector_type}, {zbb_max_multiplier}, {zc_max_multiplier})",
            params
        )
        
        # ZBollingerBandsのインスタンス化
        self._z_bb = ZBollingerBands(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=zbb_max_multiplier,
            min_multiplier=zbb_min_multiplier,
            max_std_dev_cycle_part=zbb_max_std_dev_cycle_part,
            max_std_dev_max_cycle=zbb_max_std_dev_max_cycle,
            max_std_dev_min_cycle=zbb_max_std_dev_min_cycle,
            max_std_dev_max_output=zbb_max_std_dev_max_output,
            max_std_dev_min_output=zbb_max_std_dev_min_output,
            min_std_dev_cycle_part=zbb_min_std_dev_cycle_part,
            min_std_dev_max_cycle=zbb_min_std_dev_max_cycle,
            min_std_dev_min_cycle=zbb_min_std_dev_min_cycle,
            min_std_dev_max_output=zbb_min_std_dev_max_output,
            min_std_dev_min_output=zbb_min_std_dev_min_output,
            src_type=src_type
        )
        
        # ZChannelのインスタンス化
        self._z_channel = ZChannel(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=zc_max_multiplier,
            min_multiplier=zc_min_multiplier,
            smoother_type=zc_smoother_type,
            src_type=src_type
        )
        
        # 結果キャッシュ
        self._signals = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['high', 'low', 'close']
            if 'open' in data.columns:
                cols.append('open')
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合
            data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: BBがチャネルの外側, -1: BBがチャネルの内側, 0: 判定不能)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # ZBollingerBandsの計算
            self._z_bb.calculate(data)
            
            # ZChannelの計算
            self._z_channel.calculate(data)
            
            # バンドの取得
            _, bb_upper, bb_lower = self._z_bb.get_bands()
            _, channel_upper, channel_lower = self._z_channel.get_bands()
            
            # 比較シグナルの計算（高速化版）
            signals = calculate_comparison_signals(
                bb_upper,
                bb_lower,
                channel_upper,
                channel_lower
            )
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            import traceback
            self.logger.error(f"ZBBChannelComparisonSignal計算中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                return self._signals
            else:
                return np.array([], dtype=np.int8)
    
    def get_bb_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ZBollingerBandsのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_bb.get_bands()
    
    def get_channel_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ZChannelのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_channel.get_bands()
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            self.generate(data)
            
        # Z-BBからCER値を取得
        return self._z_bb.get_cycle_er()
    
    def get_bb_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZBBの動的標準偏差乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的標準偏差乗数の値
        """
        if data is not None:
            self.generate(data)
            
        return self._z_bb.get_dynamic_multiplier()
    
    def get_channel_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZChannelの動的ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        if data is not None:
            self.generate(data)
            
        return self._z_channel.get_dynamic_multiplier()
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._z_bb.reset() if hasattr(self._z_bb, 'reset') else None
        self._z_channel.reset() if hasattr(self._z_channel, 'reset') else None
        self._signals = None
        self._data_hash = None 