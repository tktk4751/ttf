#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_bb.breakout_entry import ZBBBreakoutEntrySignal
from signals.implementations.z_trend_filter.filter import ZTrendFilterSignal


@njit(fastmath=True, parallel=True)
def calculate_combined_signals(
    breakout_signals: np.ndarray, 
    trend_signals: np.ndarray,
    lookback: int = 1
) -> np.ndarray:
    """
    ブレイクアウトシグナルとトレンドフィルターシグナルを組み合わせて
    最終的なエントリー/エグジットシグナルを計算する（高速化版）
    
    Args:
        breakout_signals: ZBBブレイクアウトシグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: シグナルなし)
        trend_signals: ZTrendフィルターシグナルの配列 (1: トレンド相場, -1: レンジ相場)
        lookback: 過去のシグナルを参照する期間
    
    Returns:
        np.ndarray: 最終的なエントリー/エグジットシグナル
    """
    length = len(breakout_signals)
    combined_signals = np.zeros(length, dtype=np.int8)
    
    # 過去lookback分のシグナルを参照してシグナルを生成（並列処理化）
    for i in prange(lookback, length):
        # 無効な値の場合は0を設定
        if np.isnan(breakout_signals[i]) or np.isnan(trend_signals[i]):
            combined_signals[i] = 0
            continue
        
        # ZBBブレイクアウトシグナルとZTrendフィルターシグナルの組み合わせ
        if breakout_signals[i] == 1 and trend_signals[i] == 1:
            # ロングエントリーシグナル
            combined_signals[i] = 1
        elif breakout_signals[i] == -1 and trend_signals[i] == 1:
            # ショートエントリーシグナル
            combined_signals[i] = -1
            
    return combined_signals


class ZBBTrendStrategySignalGenerator(BaseSignalGenerator):
    """
    ZボリンジャーバンドとZトレンドフィルターを組み合わせたシグナル生成クラス
    
    特徴:
    - ZBBBreakoutEntrySignalとZTrendFilterSignalを組み合わせたトレンドフォロー戦略
    - トレンド相場のみでエントリーシグナルを生成
    - 両シグナルのNumba最適化による高速計算
    
    シグナル条件:
    - ロングエントリー: ZBBBreakoutがロングシグナル(1)かつZTrendFilterがトレンド相場(1)
    - ショートエントリー: ZBBBreakoutがショートシグナル(-1)かつZTrendFilterがトレンド相場(1)
    - ロング決済: ZBBBreakoutがショートシグナル(-1)
    - ショート決済: ZBBBreakoutがロングシグナル(1)
    """
    
    def __init__(
        self,
        # ZBBBreakoutEntrySignalのパラメータ
        zbb_cycle_detector_type: str = 'hody_dc',
        zbb_lp_period: int = 5,
        zbb_hp_period: int = 144,
        zbb_cycle_part: float = 0.5,
        zbb_max_multiplier: float = 2.5,
        zbb_min_multiplier: float = 1.0,
        zbb_max_cycle_part: float = 0.5,
        zbb_max_max_cycle: int = 144,
        zbb_max_min_cycle: int = 10,
        zbb_max_max_output: int = 89,
        zbb_max_min_output: int = 13,
        zbb_min_cycle_part: float = 0.25,
        zbb_min_max_cycle: int = 55,
        zbb_min_min_cycle: int = 5,
        zbb_min_max_output: int = 21,
        zbb_min_min_output: int = 5,
        zbb_src_type: str = 'hlc3',
        zbb_lookback: int = 1,
        
        # ZTrendFilterSignalのパラメータ
        ztf_max_stddev_period: int = 13,
        ztf_min_stddev_period: int = 5,
        ztf_max_lookback_period: int = 13,
        ztf_min_lookback_period: int = 5,
        ztf_max_rms_window: int = 13,
        ztf_min_rms_window: int = 5,
        ztf_max_threshold: float = 0.75,
        ztf_min_threshold: float = 0.55,
        ztf_cycle_detector_type: str = 'hody_dc',
        ztf_lp_period: int = 5,
        ztf_hp_period: int = 62,
        ztf_cycle_part: float = 0.5,
        ztf_combination_weight: float = 0.6,
        ztf_zadx_weight: float = 0.4,
        ztf_combination_method: str = "sigmoid",
        ztf_max_chop_dc_cycle_part: float = 0.5,
        ztf_max_chop_dc_max_cycle: int = 144,
        ztf_max_chop_dc_min_cycle: int = 10,
        ztf_max_chop_dc_max_output: int = 34,
        ztf_max_chop_dc_min_output: int = 13,
        ztf_min_chop_dc_cycle_part: float = 0.25,
        ztf_min_chop_dc_max_cycle: int = 55,
        ztf_min_chop_dc_min_cycle: int = 5,
        ztf_min_chop_dc_max_output: int = 13,
        ztf_min_chop_dc_min_output: int = 5,
        ztf_smoother_type: str = 'alma'
    ):
        """
        コンストラクタ
        
        Args:
            zbb_cycle_detector_type: ZBBのサイクル検出器の種類
            zbb_lp_period: ZBBのローパスフィルター期間
            zbb_hp_period: ZBBのハイパスフィルター期間
            zbb_cycle_part: ZBBのサイクル部分倍率
            zbb_max_multiplier: ZBBの最大標準偏差乗数
            zbb_min_multiplier: ZBBの最小標準偏差乗数
            zbb_max_cycle_part: ZBBの最大標準偏差サイクル部分
            zbb_max_max_cycle: ZBBの最大標準偏差最大サイクル
            zbb_max_min_cycle: ZBBの最大標準偏差最小サイクル
            zbb_max_max_output: ZBBの最大標準偏差最大出力
            zbb_max_min_output: ZBBの最大標準偏差最小出力
            zbb_min_cycle_part: ZBBの最小標準偏差サイクル部分
            zbb_min_max_cycle: ZBBの最小標準偏差最大サイクル
            zbb_min_min_cycle: ZBBの最小標準偏差最小サイクル
            zbb_min_max_output: ZBBの最小標準偏差最大出力
            zbb_min_min_output: ZBBの最小標準偏差最小出力
            zbb_smoother_type: ZBBの平滑化タイプ（'alma'または'hyper'）
            zbb_hyper_smooth_period: ZBBのハイパースムーサー期間
            zbb_src_type: ZBBの価格ソースタイプ
            zbb_lookback: ZBBのルックバック期間
            
            ztf_max_stddev_period: ZTFの最大標準偏差期間
            ztf_min_stddev_period: ZTFの最小標準偏差期間
            ztf_max_lookback_period: ZTFの最大ルックバック期間
            ztf_min_lookback_period: ZTFの最小ルックバック期間
            ztf_max_rms_window: ZTFの最大RMSウィンドウ
            ztf_min_rms_window: ZTFの最小RMSウィンドウ
            ztf_max_threshold: ZTFの最大しきい値
            ztf_min_threshold: ZTFの最小しきい値
            ztf_cycle_detector_type: ZTFのサイクル検出器タイプ
            ztf_lp_period: ZTFのローパスフィルター期間
            ztf_hp_period: ZTFのハイパスフィルター期間
            ztf_cycle_part: ZTFのサイクル部分倍率
            ztf_combination_weight: ZTFの組み合わせ重み
            ztf_zadx_weight: ZTFのZADX重み
            ztf_combination_method: ZTFの組み合わせ方法
            ztf_max_chop_dc_cycle_part: ZTFの最大チョップDCサイクル部分
            ztf_max_chop_dc_max_cycle: ZTFの最大チョップDC最大サイクル
            ztf_max_chop_dc_min_cycle: ZTFの最大チョップDC最小サイクル
            ztf_max_chop_dc_max_output: ZTFの最大チョップDC最大出力
            ztf_max_chop_dc_min_output: ZTFの最大チョップDC最小出力
            ztf_min_chop_dc_cycle_part: ZTFの最小チョップDCサイクル部分
            ztf_min_chop_dc_max_cycle: ZTFの最小チョップDC最大サイクル
            ztf_min_chop_dc_min_cycle: ZTFの最小チョップDC最小サイクル
            ztf_min_chop_dc_max_output: ZTFの最小チョップDC最大出力
            ztf_min_chop_dc_min_output: ZTFの最小チョップDC最小出力
            ztf_smoother_type: ZTFの平滑化タイプ
        """
        super().__init__("ZBollingerTrendSignalGenerator")
        
        # ZBBBreakoutEntrySignalの初期化
        self.zbb_breakout = ZBBBreakoutEntrySignal(
            cycle_detector_type=zbb_cycle_detector_type,
            lp_period=zbb_lp_period,
            hp_period=zbb_hp_period,
            cycle_part=zbb_cycle_part,
            max_multiplier=zbb_max_multiplier,
            min_multiplier=zbb_min_multiplier,
            max_cycle_part=zbb_max_cycle_part,
            max_max_cycle=zbb_max_max_cycle,
            max_min_cycle=zbb_max_min_cycle,
            max_max_output=zbb_max_max_output,
            max_min_output=zbb_max_min_output,
            min_cycle_part=zbb_min_cycle_part,
            min_max_cycle=zbb_min_max_cycle,
            min_min_cycle=zbb_min_min_cycle,
            min_max_output=zbb_min_max_output,
            min_min_output=zbb_min_min_output,
            src_type=zbb_src_type,
            lookback=zbb_lookback
        )
        
        # ZTrendFilterSignalの初期化
        self.ztrend_filter = ZTrendFilterSignal(
            max_stddev_period=ztf_max_stddev_period,
            min_stddev_period=ztf_min_stddev_period,
            max_lookback_period=ztf_max_lookback_period,
            min_lookback_period=ztf_min_lookback_period,
            max_rms_window=ztf_max_rms_window,
            min_rms_window=ztf_min_rms_window,
            max_threshold=ztf_max_threshold,
            min_threshold=ztf_min_threshold,
            cycle_detector_type=ztf_cycle_detector_type,
            lp_period=ztf_lp_period,
            hp_period=ztf_hp_period,
            cycle_part=ztf_cycle_part,
            combination_weight=ztf_combination_weight,
            zadx_weight=ztf_zadx_weight,
            combination_method=ztf_combination_method,
            max_chop_dc_cycle_part=ztf_max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=ztf_max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=ztf_max_chop_dc_min_cycle,
            max_chop_dc_max_output=ztf_max_chop_dc_max_output,
            max_chop_dc_min_output=ztf_max_chop_dc_min_output,
            min_chop_dc_cycle_part=ztf_min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=ztf_min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=ztf_min_chop_dc_min_cycle,
            min_chop_dc_max_output=ztf_min_chop_dc_max_output,
            min_chop_dc_min_output=ztf_min_chop_dc_min_output,
            smoother_type=ztf_smoother_type
        )
        
        # ルックバック期間の保存
        self.lookback = zbb_lookback
        
        # キャッシュ用変数
        self._entry_signals = None
        self._combined_signals = None
        self._data_len = 0
        self._data_hash = None
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> Optional[str]:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合は必要なカラムのみハッシュする
                cols = ['open', 'high', 'low', 'close']
                return str(hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns)))))
            else:
                # NumPy配列の場合は全体をハッシュする
                return str(hash(tuple(map(tuple, data))))
        except Exception:
            return None
        
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            np.ndarray: エントリーシグナル (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # データのサイズ変更を検出
            data_len = len(data)
            data_hash = self._get_data_hash(data)
            
            # キャッシュが有効な場合はキャッシュを返す
            if data_hash == self._data_hash and data_len == self._data_len and self._entry_signals is not None:
                return self._entry_signals
            
            # キャッシュの更新
            self._data_len = data_len
            self._data_hash = data_hash
            
            # ZBBのブレイクアウトシグナルを取得
            breakout_signals = self.zbb_breakout.generate(data)
            
            # ZTrendのフィルターシグナルを取得
            trend_signals = self.ztrend_filter.generate(data)
            
            # 組み合わせたシグナルを計算
            combined_signals = calculate_combined_signals(
                breakout_signals, 
                trend_signals,
                self.lookback
            )
            
            # 結果をキャッシュして返す
            self._combined_signals = combined_signals
            self._entry_signals = combined_signals
            return self._entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            import traceback
            self.logger.error(f"ZBollingerTrendSignalGenerator計算中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.zeros(len(data), dtype=np.int8)
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを取得する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべき場合はTrue
        """
        try:
            # エントリーシグナルを事前に計算（キャッシュも行う）
            if self._entry_signals is None or len(self._entry_signals) != len(data):
                self.get_entry_signals(data)
                
            # ZBBのブレイクアウトシグナルを取得
            breakout_signals = self.zbb_breakout.generate(data)
            
            # 指定されたインデックスのブレイクアウトシグナルを取得
            current_signal = breakout_signals[index]
            
            # ポジションに基づいてエグジット条件を判定
            if position == 1:  # ロングポジション
                # ショートシグナルが出た場合にエグジット
                return current_signal == -1
            elif position == -1:  # ショートポジション
                # ロングシグナルが出た場合にエグジット
                return current_signal == 1
            
            return False
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、エグジットしないと判断
            import traceback
            self.logger.error(f"ZBollingerTrendSignalGenerator エグジット計算中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        Zボリンジャーバンドのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.zbb_breakout.generate(data)
            
        return self.zbb_breakout.get_bands()
    
    def get_trend_filter(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドフィルターの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンドフィルター値
        """
        if data is not None:
            self.ztrend_filter.generate(data)
            
        return self.ztrend_filter.get_filter_values()
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            self.zbb_breakout.generate(data)
            
        return self.zbb_breakout.get_cycle_er()
    
    def reset(self) -> None:
        """
        シグナル生成器の状態をリセットする
        """
        self.zbb_breakout.reset() if hasattr(self.zbb_breakout, 'reset') else None
        self.ztrend_filter.reset() if hasattr(self.ztrend_filter, 'reset') else None
        self._entry_signals = None
        self._combined_signals = None
        self._data_len = 0
        self._data_hash = None 