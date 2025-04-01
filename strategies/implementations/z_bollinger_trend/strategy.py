#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZBBTrendStrategySignalGenerator


class ZBBTrendStrategy(BaseStrategy):
    """
    ZボリンジャーバンドとZトレンドフィルターを組み合わせたトレンドフォロー戦略
    
    特徴:
    - ZBBBreakoutEntrySignalとZTrendFilterSignalを組み合わせたトレンドフォロー戦略
    - トレンド相場のみでエントリーシグナルを生成
    - 両シグナルのNumba最適化による高速計算
    - ブレイクアウトシグナルによる自動決済
    
    エントリー条件:
    - ロングエントリー: ZBBBreakoutがロングシグナル(1)かつZTrendFilterがトレンド相場(1)
    - ショートエントリー: ZBBBreakoutがショートシグナル(-1)かつZTrendFilterがトレンド相場(1)
    
    エグジット条件:
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
            
            stop_atr_period: ストップロス計算用のATR期間
            stop_atr_multiplier: ストップロス計算用のATR乗数
        """
        super().__init__("ZBollingerTrend")
        
        # 戦略パラメータの設定
        self._parameters = {
            'zbb_cycle_detector_type': zbb_cycle_detector_type,
            'zbb_lp_period': zbb_lp_period,
            'zbb_hp_period': zbb_hp_period,
            'zbb_cycle_part': zbb_cycle_part,
            'zbb_max_multiplier': zbb_max_multiplier,
            'zbb_min_multiplier': zbb_min_multiplier,
            'zbb_max_cycle_part': zbb_max_cycle_part,
            'zbb_max_max_cycle': zbb_max_max_cycle,
            'zbb_max_min_cycle': zbb_max_min_cycle,
            'zbb_max_max_output': zbb_max_max_output,
            'zbb_max_min_output': zbb_max_min_output,
            'zbb_min_cycle_part': zbb_min_cycle_part,
            'zbb_min_max_cycle': zbb_min_max_cycle,
            'zbb_min_min_cycle': zbb_min_min_cycle,
            'zbb_min_max_output': zbb_min_max_output,
            'zbb_min_min_output': zbb_min_min_output,
            'zbb_src_type': zbb_src_type,
            'zbb_lookback': zbb_lookback,
            'ztf_max_stddev_period': ztf_max_stddev_period,
            'ztf_min_stddev_period': ztf_min_stddev_period,
            'ztf_max_lookback_period': ztf_max_lookback_period,
            'ztf_min_lookback_period': ztf_min_lookback_period,
            'ztf_max_rms_window': ztf_max_rms_window,
            'ztf_min_rms_window': ztf_min_rms_window,
            'ztf_max_threshold': ztf_max_threshold,
            'ztf_min_threshold': ztf_min_threshold,
            'ztf_cycle_detector_type': ztf_cycle_detector_type,
            'ztf_lp_period': ztf_lp_period,
            'ztf_hp_period': ztf_hp_period,
            'ztf_cycle_part': ztf_cycle_part,
            'ztf_combination_weight': ztf_combination_weight,
            'ztf_zadx_weight': ztf_zadx_weight,
            'ztf_combination_method': ztf_combination_method,
            'ztf_max_chop_dc_cycle_part': ztf_max_chop_dc_cycle_part,
            'ztf_max_chop_dc_max_cycle': ztf_max_chop_dc_max_cycle,
            'ztf_max_chop_dc_min_cycle': ztf_max_chop_dc_min_cycle,
            'ztf_max_chop_dc_max_output': ztf_max_chop_dc_max_output,
            'ztf_max_chop_dc_min_output': ztf_max_chop_dc_min_output,
            'ztf_min_chop_dc_cycle_part': ztf_min_chop_dc_cycle_part,
            'ztf_min_chop_dc_max_cycle': ztf_min_chop_dc_max_cycle,
            'ztf_min_chop_dc_min_cycle': ztf_min_chop_dc_min_cycle,
            'ztf_min_chop_dc_max_output': ztf_min_chop_dc_max_output,
            'ztf_min_chop_dc_min_output': ztf_min_chop_dc_min_output,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZBBTrendStrategySignalGenerator(
            zbb_cycle_detector_type=zbb_cycle_detector_type,
            zbb_lp_period=zbb_lp_period,
            zbb_hp_period=zbb_hp_period,
            zbb_cycle_part=zbb_cycle_part,
            zbb_max_multiplier=zbb_max_multiplier,
            zbb_min_multiplier=zbb_min_multiplier,
            zbb_max_cycle_part=zbb_max_cycle_part,
            zbb_max_max_cycle=zbb_max_max_cycle,
            zbb_max_min_cycle=zbb_max_min_cycle,
            zbb_max_max_output=zbb_max_max_output,
            zbb_max_min_output=zbb_max_min_output,
            zbb_min_cycle_part=zbb_min_cycle_part,
            zbb_min_max_cycle=zbb_min_max_cycle,
            zbb_min_min_cycle=zbb_min_min_cycle,
            zbb_min_max_output=zbb_min_max_output,
            zbb_min_min_output=zbb_min_min_output,
            zbb_src_type=zbb_src_type,
            zbb_lookback=zbb_lookback,
            ztf_max_stddev_period=ztf_max_stddev_period,
            ztf_min_stddev_period=ztf_min_stddev_period,
            ztf_max_lookback_period=ztf_max_lookback_period,
            ztf_min_lookback_period=ztf_min_lookback_period,
            ztf_max_rms_window=ztf_max_rms_window,
            ztf_min_rms_window=ztf_min_rms_window,
            ztf_max_threshold=ztf_max_threshold,
            ztf_min_threshold=ztf_min_threshold,
            ztf_cycle_detector_type=ztf_cycle_detector_type,
            ztf_lp_period=ztf_lp_period,
            ztf_hp_period=ztf_hp_period,
            ztf_cycle_part=ztf_cycle_part,
            ztf_combination_weight=ztf_combination_weight,
            ztf_zadx_weight=ztf_zadx_weight,
            ztf_combination_method=ztf_combination_method,
            ztf_max_chop_dc_cycle_part=ztf_max_chop_dc_cycle_part,
            ztf_max_chop_dc_max_cycle=ztf_max_chop_dc_max_cycle,
            ztf_max_chop_dc_min_cycle=ztf_max_chop_dc_min_cycle,
            ztf_max_chop_dc_max_output=ztf_max_chop_dc_max_output,
            ztf_max_chop_dc_min_output=ztf_max_chop_dc_min_output,
            ztf_min_chop_dc_cycle_part=ztf_min_chop_dc_cycle_part,
            ztf_min_chop_dc_max_cycle=ztf_min_chop_dc_max_cycle,
            ztf_min_chop_dc_min_cycle=ztf_min_chop_dc_min_cycle,
            ztf_min_chop_dc_max_output=ztf_min_chop_dc_max_output,
            ztf_min_chop_dc_min_output=ztf_min_chop_dc_min_output,
        )
        
        # ATRの計算用キャッシュ
        self._atr_values = None
        self._atr_data_len = 0
        
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        return self.signal_generator.get_exit_signals(data, position, index)
    
    def _calculate_atr(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ATR（Average True Range）を計算する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: ATR値
        """
        # キャッシュが有効ならそれを使用
        if self._atr_values is not None and len(data) == self._atr_data_len:
            return self._atr_values
        
        # データの準備
        if isinstance(data, pd.DataFrame):
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
        else:
            high = data[:, 1]
            low = data[:, 2]
            close = data[:, 3]
        
        # ATRの計算
        period = self._parameters['stop_atr_period']
        tr = np.zeros(len(close))
        
        # 最初の行のTRは高値と安値の差
        tr[0] = high[0] - low[0]
        
        # 2番目以降のTRを計算
        for i in range(1, len(close)):
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            hl = high[i] - low[i]
            hpc = abs(high[i] - close[i-1])
            lpc = abs(low[i] - close[i-1])
            tr[i] = max(hl, max(hpc, lpc))
        
        # 指数移動平均でATRを計算
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])
        
        # EMAの更新
        multiplier = 2.0 / (period + 1)
        for i in range(period, len(tr)):
            atr[i] = (tr[i] * multiplier) + (atr[i-1] * (1.0 - multiplier))
        
        # キャッシュを更新
        self._atr_values = atr
        self._atr_data_len = len(data)
        
        return atr
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            float: ストップロス価格
        """
        # ATRの計算
        atr = self._calculate_atr(data)
        
        # データの準備
        if isinstance(data, pd.DataFrame):
            close = data['close'].iloc[index]
        else:
            close = data[index, 3]
        
        # ATR値の取得（無効な値の場合はデフォルト値を使用）
        if index < 0:
            index = len(atr) + index
        
        atr_value = atr[index]
        if np.isnan(atr_value) or atr_value <= 0:
            atr_value = close * 0.01  # ATRがNaNまたは0以下の場合は価格の1%をデフォルト値として使用
        
        # ATRに基づくストップロス価格の計算
        atr_multiplier = self._parameters['stop_atr_multiplier']
        
        if position == 1:  # ロングポジション
            return close - (atr_value * atr_multiplier)
        else:  # ショートポジション
            return close + (atr_value * atr_multiplier)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成する
        
        Args:
            trial: Optunaのtrialオブジェクト
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            # ZBBパラメータ
            'zbb_cycle_detector_type': trial.suggest_categorical('zbb_cycle_detector_type', ['hody_dc', 'dudi_dc', 'phac_dc']),
            'zbb_max_multiplier': trial.suggest_float('zbb_max_multiplier', 1.5, 3.0, step=0.1),
            'zbb_min_multiplier': trial.suggest_float('zbb_min_multiplier', 0.5, 1.5, step=0.1),
            'zbb_hyper_smooth_period': trial.suggest_int('zbb_hyper_smooth_period', 0, 8),
            'zbb_lookback': trial.suggest_int('zbb_lookback', 1, 5),
            
            # ZTFパラメータ
            'ztf_max_threshold': trial.suggest_float('ztf_max_threshold', 0.6, 0.9, step=0.05),
            'ztf_min_threshold': trial.suggest_float('ztf_min_threshold', 0.3, 0.6, step=0.05),
            'ztf_combination_weight': trial.suggest_float('ztf_combination_weight', 0.3, 0.8, step=0.1),
            'ztf_zadx_weight': trial.suggest_float('ztf_zadx_weight', 0.2, 0.7, step=0.1),
            'ztf_combination_method': trial.suggest_categorical('ztf_combination_method', ['sigmoid', 'rms', 'simple']),
            
        }
        
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換する
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        # 全パラメータをコピー
        strategy_params = params.copy()
        
        return strategy_params 