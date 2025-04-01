#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZMeanReversionSignalGenerator


class ZMeanReversionStrategy(BaseStrategy):
    """
    Z平均回帰戦略
    
    戦略の特徴:
    - レンジ相場特化型の平均回帰戦略
    - ZRSXトリガーとZハーストエクスポネントを組み合わせてエントリー
    - Zドンチャンブレイクアウトによる利益確定
    
    エントリー条件:
    - ロング: ZRSXが買いシグナル(1)かつZハーストがレンジ相場(-1)
    - ショート: ZRSXが売りシグナル(-1)かつZハーストがレンジ相場(-1)
    
    エグジット条件:
    - ロング決済: Zドンチャンが売りシグナル(-1)
    - ショート決済: Zドンチャンが買いシグナル(1)
    """
    
    def __init__(
        self,
        # ZRSXトリガーシグナルのパラメータ
        # サイクル効率比(ER)のパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        er_period: int = 10,
        
        # 最大ドミナントサイクル計算パラメータ
        rsx_max_dc_cycle_part: float = 0.5,
        rsx_max_dc_max_cycle: int = 55,
        rsx_max_dc_min_cycle: int = 5,
        rsx_max_dc_max_output: int = 34,
        rsx_max_dc_min_output: int = 14,
        
        # 最小ドミナントサイクル計算パラメータ
        rsx_min_dc_cycle_part: float = 0.25,
        rsx_min_dc_max_cycle: int = 34,
        rsx_min_dc_min_cycle: int = 3,
        rsx_min_dc_max_output: int = 13,
        rsx_min_dc_min_output: int = 3,
        
        # 買われすぎ/売られすぎレベルパラメータ
        min_high_level: float = 75.0,
        max_high_level: float = 85.0,
        min_low_level: float = 15.0,
        max_low_level: float = 25.0,
        
        # Zハーストエクスポネントシグナルのパラメータ
        # 分析ウィンドウパラメータ
        hurst_max_window_dc_cycle_part: float = 0.75,
        hurst_max_window_dc_max_cycle: int = 144,
        hurst_max_window_dc_min_cycle: int = 8,
        hurst_max_window_dc_max_output: int = 200,
        hurst_max_window_dc_min_output: int = 50,
        
        hurst_min_window_dc_cycle_part: float = 0.5,
        hurst_min_window_dc_max_cycle: int = 55,
        hurst_min_window_dc_min_cycle: int = 5,
        hurst_min_window_dc_max_output: int = 50,
        hurst_min_window_dc_min_output: int = 20,
        
        # ラグパラメータ
        max_lag_ratio: float = 0.5,
        min_lag_ratio: float = 0.1,
        
        # ステップパラメータ
        max_step: int = 10,
        min_step: int = 2,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.7,
        min_threshold: float = 0.55,
        
        # Zドンチャンブレイクアウトのパラメータ
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
        lookback: int = 1,
        
        # ソースタイプ
        src_type: str = 'hlc3'
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 13）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            er_period: 効率比の計算期間（デフォルト: 10）
            
            rsx_max_dc_cycle_part: RSX最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            rsx_max_dc_max_cycle: RSX最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            rsx_max_dc_min_cycle: RSX最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            rsx_max_dc_max_output: RSX最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 34）
            rsx_max_dc_min_output: RSX最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 14）
            
            rsx_min_dc_cycle_part: RSX最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            rsx_min_dc_max_cycle: RSX最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 34）
            rsx_min_dc_min_cycle: RSX最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 3）
            rsx_min_dc_max_output: RSX最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            rsx_min_dc_min_output: RSX最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            min_high_level: 最小買われすぎレベル（デフォルト: 75.0）
            max_high_level: 最大買われすぎレベル（デフォルト: 85.0）
            min_low_level: 最小売られすぎレベル（デフォルト: 25.0）
            max_low_level: 最大売られすぎレベル（デフォルト: 15.0）
            
            hurst_max_window_dc_cycle_part: ハースト最大ウィンドウ用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.75）
            hurst_max_window_dc_max_cycle: ハースト最大ウィンドウ用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            hurst_max_window_dc_min_cycle: ハースト最大ウィンドウ用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 8）
            hurst_max_window_dc_max_output: ハースト最大ウィンドウ用ドミナントサイクル計算用の最大出力値（デフォルト: 200）
            hurst_max_window_dc_min_output: ハースト最大ウィンドウ用ドミナントサイクル計算用の最小出力値（デフォルト: 50）
            
            hurst_min_window_dc_cycle_part: ハースト最小ウィンドウ用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            hurst_min_window_dc_max_cycle: ハースト最小ウィンドウ用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            hurst_min_window_dc_min_cycle: ハースト最小ウィンドウ用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            hurst_min_window_dc_max_output: ハースト最小ウィンドウ用ドミナントサイクル計算用の最大出力値（デフォルト: 50）
            hurst_min_window_dc_min_output: ハースト最小ウィンドウ用ドミナントサイクル計算用の最小出力値（デフォルト: 20）
            
            max_lag_ratio: 最大ラグとウィンドウの比率（デフォルト: 0.5）
            min_lag_ratio: 最小ラグとウィンドウの比率（デフォルト: 0.1）
            
            max_step: 最大ステップ（デフォルト: 10）
            min_step: 最小ステップ（デフォルト: 2）
            
            max_threshold: 最大しきい値（デフォルト: 0.7）
            min_threshold: 最小しきい値（デフォルト: 0.55）
            
            donchian_max_dc_cycle_part: ドンチャン最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            donchian_max_dc_max_cycle: ドンチャン最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            donchian_max_dc_min_cycle: ドンチャン最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 13）
            donchian_max_dc_max_output: ドンチャン最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 89）
            donchian_max_dc_min_output: ドンチャン最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 21）
            
            donchian_min_dc_cycle_part: ドンチャン最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            donchian_min_dc_max_cycle: ドンチャン最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            donchian_min_dc_min_cycle: ドンチャン最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            donchian_min_dc_max_output: ドンチャン最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 21）
            donchian_min_dc_min_output: ドンチャン最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 8）
            
            lookback: 過去バンド参照期間（デフォルト: 1）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
        """
        # 戦略名の設定
        strategy_name = f"ZMeanReversion({cycle_detector_type}, H:{min_threshold}-{max_threshold}, RSX:{min_low_level}-{max_high_level})"
        super().__init__(strategy_name)
        
        # パラメータの設定
        self._params = {
            # ZRSXトリガーパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'er_period': er_period,
            
            'rsx_max_dc_cycle_part': rsx_max_dc_cycle_part,
            'rsx_max_dc_max_cycle': rsx_max_dc_max_cycle,
            'rsx_max_dc_min_cycle': rsx_max_dc_min_cycle,
            'rsx_max_dc_max_output': rsx_max_dc_max_output,
            'rsx_max_dc_min_output': rsx_max_dc_min_output,
            
            'rsx_min_dc_cycle_part': rsx_min_dc_cycle_part,
            'rsx_min_dc_max_cycle': rsx_min_dc_max_cycle,
            'rsx_min_dc_min_cycle': rsx_min_dc_min_cycle,
            'rsx_min_dc_max_output': rsx_min_dc_max_output,
            'rsx_min_dc_min_output': rsx_min_dc_min_output,
            
            'min_high_level': min_high_level,
            'max_high_level': max_high_level,
            'min_low_level': min_low_level,
            'max_low_level': max_low_level,
            
            # Zハーストパラメータ
            'hurst_max_window_dc_cycle_part': hurst_max_window_dc_cycle_part,
            'hurst_max_window_dc_max_cycle': hurst_max_window_dc_max_cycle,
            'hurst_max_window_dc_min_cycle': hurst_max_window_dc_min_cycle,
            'hurst_max_window_dc_max_output': hurst_max_window_dc_max_output,
            'hurst_max_window_dc_min_output': hurst_max_window_dc_min_output,
            
            'hurst_min_window_dc_cycle_part': hurst_min_window_dc_cycle_part,
            'hurst_min_window_dc_max_cycle': hurst_min_window_dc_max_cycle,
            'hurst_min_window_dc_min_cycle': hurst_min_window_dc_min_cycle,
            'hurst_min_window_dc_max_output': hurst_min_window_dc_max_output,
            'hurst_min_window_dc_min_output': hurst_min_window_dc_min_output,
            
            'max_lag_ratio': max_lag_ratio,
            'min_lag_ratio': min_lag_ratio,
            
            'max_step': max_step,
            'min_step': min_step,
            
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            
            # Zドンチャンパラメータ
            'donchian_max_dc_cycle_part': donchian_max_dc_cycle_part,
            'donchian_max_dc_max_cycle': donchian_max_dc_max_cycle,
            'donchian_max_dc_min_cycle': donchian_max_dc_min_cycle,
            'donchian_max_dc_max_output': donchian_max_dc_max_output,
            'donchian_max_dc_min_output': donchian_max_dc_min_output,
            
            'donchian_min_dc_cycle_part': donchian_min_dc_cycle_part,
            'donchian_min_dc_max_cycle': donchian_min_dc_max_cycle,
            'donchian_min_dc_min_cycle': donchian_min_dc_min_cycle,
            'donchian_min_dc_max_output': donchian_min_dc_max_output,
            'donchian_min_dc_min_output': donchian_min_dc_min_output,
            
            'lookback': lookback,
            'src_type': src_type
        }
        
        # シグナル生成器の初期化
        self._signal_generator = ZMeanReversionSignalGenerator(
            # ZRSXトリガーパラメータ
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            er_period=er_period,
            
            rsx_max_dc_cycle_part=rsx_max_dc_cycle_part,
            rsx_max_dc_max_cycle=rsx_max_dc_max_cycle,
            rsx_max_dc_min_cycle=rsx_max_dc_min_cycle,
            rsx_max_dc_max_output=rsx_max_dc_max_output,
            rsx_max_dc_min_output=rsx_max_dc_min_output,
            
            rsx_min_dc_cycle_part=rsx_min_dc_cycle_part,
            rsx_min_dc_max_cycle=rsx_min_dc_max_cycle,
            rsx_min_dc_min_cycle=rsx_min_dc_min_cycle,
            rsx_min_dc_max_output=rsx_min_dc_max_output,
            rsx_min_dc_min_output=rsx_min_dc_min_output,
            
            min_high_level=min_high_level,
            max_high_level=max_high_level,
            min_low_level=min_low_level,
            max_low_level=max_low_level,
            
            # Zハーストパラメータ
            hurst_max_window_dc_cycle_part=hurst_max_window_dc_cycle_part,
            hurst_max_window_dc_max_cycle=hurst_max_window_dc_max_cycle,
            hurst_max_window_dc_min_cycle=hurst_max_window_dc_min_cycle,
            hurst_max_window_dc_max_output=hurst_max_window_dc_max_output,
            hurst_max_window_dc_min_output=hurst_max_window_dc_min_output,
            
            hurst_min_window_dc_cycle_part=hurst_min_window_dc_cycle_part,
            hurst_min_window_dc_max_cycle=hurst_min_window_dc_max_cycle,
            hurst_min_window_dc_min_cycle=hurst_min_window_dc_min_cycle,
            hurst_min_window_dc_max_output=hurst_min_window_dc_max_output,
            hurst_min_window_dc_min_output=hurst_min_window_dc_min_output,
            
            max_lag_ratio=max_lag_ratio,
            min_lag_ratio=min_lag_ratio,
            
            max_step=max_step,
            min_step=min_step,
            
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            
            # Zドンチャンパラメータ
            donchian_max_dc_cycle_part=donchian_max_dc_cycle_part,
            donchian_max_dc_max_cycle=donchian_max_dc_max_cycle,
            donchian_max_dc_min_cycle=donchian_max_dc_min_cycle,
            donchian_max_dc_max_output=donchian_max_dc_max_output,
            donchian_max_dc_min_output=donchian_max_dc_min_output,
            
            donchian_min_dc_cycle_part=donchian_min_dc_cycle_part,
            donchian_min_dc_max_cycle=donchian_min_dc_max_cycle,
            donchian_min_dc_min_cycle=donchian_min_dc_min_cycle,
            donchian_min_dc_max_output=donchian_min_dc_max_output,
            donchian_min_dc_min_output=donchian_min_dc_min_output,
            
            lookback=lookback,
            src_type=src_type
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            np.ndarray: エントリーシグナル配列
        """
        try:
            return self._signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング, -1: ショート）
            index: シグナルを確認するインデックス（デフォルト: -1（最新））
        
        Returns:
            bool: エグジットすべきかどうか
        """
        try:
            return self._signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成する
        
        Args:
            trial: optunaのtrialオブジェクト
            
        Returns:
            Dict[str, Any]: 最適化パラメータの辞書
        """
        params = {
            # サイクル検出器タイプ
            'cycle_detector_type': trial.suggest_categorical(
                'cycle_detector_type', 
                ['dudi_dce', 'phac_dce', 'hody_dc']
            ),
            
            # 効率比とサイクル部分の基本パラメータ
            'lp_period': trial.suggest_int('lp_period', 5, 21),
            'hp_period': trial.suggest_int('hp_period', 34, 200),
            'cycle_part': trial.suggest_float('cycle_part', 0.25, 0.75),
            'er_period': trial.suggest_int('er_period', 5, 21),
            
            # RSXパラメータ
            # 最大期間用
            'rsx_max_dc_cycle_part': trial.suggest_float('rsx_max_dc_cycle_part', 0.4, 0.8),
            'rsx_max_dc_max_cycle': trial.suggest_int('rsx_max_dc_max_cycle', 34, 89),
            'rsx_max_dc_min_cycle': trial.suggest_int('rsx_max_dc_min_cycle', 3, 13),
            'rsx_max_dc_max_output': trial.suggest_int('rsx_max_dc_max_output', 21, 55),
            'rsx_max_dc_min_output': trial.suggest_int('rsx_max_dc_min_output', 8, 21),
            
            # 最小期間用
            'rsx_min_dc_cycle_part': trial.suggest_float('rsx_min_dc_cycle_part', 0.2, 0.5),
            'rsx_min_dc_max_cycle': trial.suggest_int('rsx_min_dc_max_cycle', 21, 55),
            'rsx_min_dc_min_cycle': trial.suggest_int('rsx_min_dc_min_cycle', 2, 8),
            'rsx_min_dc_max_output': trial.suggest_int('rsx_min_dc_max_output', 8, 21),
            'rsx_min_dc_min_output': trial.suggest_int('rsx_min_dc_min_output', 2, 8),
            
            # 買われすぎ/売られすぎレベル
            'min_high_level': trial.suggest_float('min_high_level', 65.0, 80.0),
            'max_high_level': trial.suggest_float('max_high_level', 80.0, 90.0),
            'min_low_level': trial.suggest_float('min_low_level', 20.0, 35.0),
            'max_low_level': trial.suggest_float('max_low_level', 10.0, 20.0),
            
            # ハーストパラメータ
            # 最大ウィンドウ用
            'hurst_max_window_dc_cycle_part': trial.suggest_float('hurst_max_window_dc_cycle_part', 0.5, 0.9),
            'hurst_max_window_dc_max_cycle': trial.suggest_int('hurst_max_window_dc_max_cycle', 89, 200),
            'hurst_max_window_dc_min_cycle': trial.suggest_int('hurst_max_window_dc_min_cycle', 5, 13),
            'hurst_max_window_dc_max_output': trial.suggest_int('hurst_max_window_dc_max_output', 100, 250),
            'hurst_max_window_dc_min_output': trial.suggest_int('hurst_max_window_dc_min_output', 34, 89),
            
            # 最小ウィンドウ用
            'hurst_min_window_dc_cycle_part': trial.suggest_float('hurst_min_window_dc_cycle_part', 0.4, 0.6),
            'hurst_min_window_dc_max_cycle': trial.suggest_int('hurst_min_window_dc_max_cycle', 34, 89),
            'hurst_min_window_dc_min_cycle': trial.suggest_int('hurst_min_window_dc_min_cycle', 3, 8),
            'hurst_min_window_dc_max_output': trial.suggest_int('hurst_min_window_dc_max_output', 34, 89),
            'hurst_min_window_dc_min_output': trial.suggest_int('hurst_min_window_dc_min_output', 13, 34),
            
            # ラグとステップ
            'max_lag_ratio': trial.suggest_float('max_lag_ratio', 0.3, 0.7),
            'min_lag_ratio': trial.suggest_float('min_lag_ratio', 0.05, 0.2),
            'max_step': trial.suggest_int('max_step', 5, 15),
            'min_step': trial.suggest_int('min_step', 1, 4),
            
            # ハーストしきい値
            'max_threshold': trial.suggest_float('max_threshold', 0.65, 0.8),
            'min_threshold': trial.suggest_float('min_threshold', 0.5, 0.65),
            
            # ドンチャンパラメータ
            # 最大期間用
            'donchian_max_dc_cycle_part': trial.suggest_float('donchian_max_dc_cycle_part', 0.4, 0.8),
            'donchian_max_dc_max_cycle': trial.suggest_int('donchian_max_dc_max_cycle', 89, 200),
            'donchian_max_dc_min_cycle': trial.suggest_int('donchian_max_dc_min_cycle', 8, 21),
            'donchian_max_dc_max_output': trial.suggest_int('donchian_max_dc_max_output', 55, 120),
            'donchian_max_dc_min_output': trial.suggest_int('donchian_max_dc_min_output', 13, 34),
            
            # 最小期間用
            'donchian_min_dc_cycle_part': trial.suggest_float('donchian_min_dc_cycle_part', 0.2, 0.5),
            'donchian_min_dc_max_cycle': trial.suggest_int('donchian_min_dc_max_cycle', 34, 89),
            'donchian_min_dc_min_cycle': trial.suggest_int('donchian_min_dc_min_cycle', 3, 8),
            'donchian_min_dc_max_output': trial.suggest_int('donchian_min_dc_max_output', 13, 34),
            'donchian_min_dc_min_output': trial.suggest_int('donchian_min_dc_min_output', 5, 13),
            
            # ブレイクアウトとソース
            'lookback': trial.suggest_int('lookback', 1, 3),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'ohlc4'])
        }
        
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略フォーマットに変換する
        
        Args:
            params: 最適化パラメータの辞書
        
        Returns:
            Dict[str, Any]: 戦略フォーマットのパラメータ辞書
        """
        # パラメータをそのまま返す（このクラスでは変換は不要）
        return params
    
    def get_additional_metrics(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        追加のメトリクスを取得する
        
        Args:
            data: 価格データ
        
        Returns:
            Dict[str, Any]: 追加メトリクスの辞書
        """
        try:
            # データを準備
            if self._signal_generator._signals is None or len(data) != self._signal_generator._data_len:
                self._signal_generator.calculate_signals(data)
            
            # RSX値と適応的レベル
            rsx_values = self._signal_generator.get_rsx_values()
            high_levels, low_levels = self._signal_generator.get_rsx_levels()
            
            # ハースト値としきい値
            hurst_values = self._signal_generator.get_hurst_values()
            hurst_threshold = self._signal_generator.get_hurst_threshold()
            
            # ドンチャンバンド
            upper, lower, middle = self._signal_generator.get_donchian_bands()
            
            # 効率比
            er_values = self._signal_generator.get_efficiency_ratio()
            
            # 最終インデックスの値を取得
            idx = len(data) - 1
            metrics = {
                'rsx_value': float(rsx_values[idx]) if len(rsx_values) > idx else None,
                'rsx_high_level': float(high_levels[idx]) if len(high_levels) > idx else None,
                'rsx_low_level': float(low_levels[idx]) if len(low_levels) > idx else None,
                'hurst_value': float(hurst_values[idx]) if len(hurst_values) > idx else None,
                'hurst_threshold': float(hurst_threshold[idx]) if len(hurst_threshold) > idx else None,
                'donchian_upper': float(upper[idx]) if len(upper) > idx else None,
                'donchian_lower': float(lower[idx]) if len(lower) > idx else None,
                'donchian_middle': float(middle[idx]) if len(middle) > idx else None,
                'efficiency_ratio': float(er_values[idx]) if len(er_values) > idx else None,
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"追加メトリクス取得中にエラー: {str(e)}")
            return {} 