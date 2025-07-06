#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import UltimateChopTrendSignalGenerator


class UltimateChopTrendStrategy(BaseStrategy):
    """
    Ultimate Chop Trend V3ストラテジー
    
    特徴:
    - Ultimate Chop Trend V3の5つのアルゴリズム統合による高精度なトレンド分析
    - トレンド方向の転換点を検出してエントリー/決済シグナルを生成
    - 信頼度による品質フィルタリング
    - Numbaによる高速化処理
    
    エントリー条件:
    - ロング: trend_direction が -1 から 1 に変化 + 信頼度条件
    - ショート: trend_direction が 1 から -1 に変化 + 信頼度条件
    
    エグジット条件:
    - 通常: トレンド方向が反転した時
    - 強制: トレンド方向が反転またはレンジになった時
    """
    
    def __init__(
        self,
        # Ultimate Chop Trend V3パラメータ
        analysis_period: int = 14,
        fast_period: int = 7,
        trend_threshold: float = 0.58,
        confidence_threshold: float = 0.3,
        
        # アルゴリズム有効化（全て軽量で効果的）
        enable_hilbert: bool = True,
        enable_regression: bool = True,
        enable_consensus: bool = True,
        enable_volatility: bool = True,
        enable_zerollag: bool = True,
        
        # シグナル生成パラメータ
        enable_exit_signals: bool = True,
        min_confidence_for_entry: float = 0.4,
        require_strong_confidence: bool = True,
        use_strong_exit: bool = False,
        confidence_filter_strength: float = 0.3
    ):
        """
        初期化
        
        Args:
            # Ultimate Chop Trend V3パラメータ
            analysis_period: 分析期間（デフォルト: 14）
            fast_period: 高速期間（デフォルト: 7）
            trend_threshold: トレンド判定しきい値（デフォルト: 0.58）
            confidence_threshold: 信頼度しきい値（デフォルト: 0.3）
            
            # アルゴリズム有効化
            enable_hilbert: ヒルベルト変換を有効にするか
            enable_regression: 増分回帰を有効にするか
            enable_consensus: コンセンサスを有効にするか
            enable_volatility: ボラティリティ分析を有効にするか
            enable_zerollag: ゼロラグEMAを有効にするか
            
            # シグナル生成パラメータ
            enable_exit_signals: 決済シグナルを有効にするか
            min_confidence_for_entry: エントリーに必要な最小信頼度
            require_strong_confidence: 強い信頼度を要求するか
            use_strong_exit: 強い決済条件を使用するか
            confidence_filter_strength: 信頼度フィルター強度
        """
        super().__init__("UltimateChopTrend")
        
        # パラメータの設定
        self._parameters = {
            'analysis_period': analysis_period,
            'fast_period': fast_period,
            'trend_threshold': trend_threshold,
            'confidence_threshold': confidence_threshold,
            'enable_hilbert': enable_hilbert,
            'enable_regression': enable_regression,
            'enable_consensus': enable_consensus,
            'enable_volatility': enable_volatility,
            'enable_zerollag': enable_zerollag,
            'enable_exit_signals': enable_exit_signals,
            'min_confidence_for_entry': min_confidence_for_entry,
            'require_strong_confidence': require_strong_confidence,
            'use_strong_exit': use_strong_exit,
            'confidence_filter_strength': confidence_filter_strength
        }
        
        # シグナル生成器の初期化
        self.signal_generator = UltimateChopTrendSignalGenerator(
            analysis_period=analysis_period,
            fast_period=fast_period,
            trend_threshold=trend_threshold,
            confidence_threshold=confidence_threshold,
            enable_hilbert=enable_hilbert,
            enable_regression=enable_regression,
            enable_consensus=enable_consensus,
            enable_volatility=enable_volatility,
            enable_zerollag=enable_zerollag,
            enable_exit_signals=enable_exit_signals,
            min_confidence_for_entry=min_confidence_for_entry,
            require_strong_confidence=require_strong_confidence,
            use_strong_exit=use_strong_exit,
            confidence_filter_strength=confidence_filter_strength
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（1=ロング、-1=ショート、0=シグナルなし）
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
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
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    def get_trend_direction(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        トレンド方向を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=レンジ）
        """
        try:
            return self.signal_generator.get_trend_direction(data)
        except Exception as e:
            self.logger.error(f"トレンド方向取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_index(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        統合トレンド指数を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 統合トレンド指数の配列（0-1）
        """
        try:
            return self.signal_generator.get_trend_index(data)
        except Exception as e:
            self.logger.error(f"統合トレンド指数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_strength(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        トレンド強度を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: トレンド強度の配列（0-1）
        """
        try:
            return self.signal_generator.get_trend_strength(data)
        except Exception as e:
            self.logger.error(f"トレンド強度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_confidence_score(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測信頼度を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 予測信頼度の配列（0-1）
        """
        try:
            return self.signal_generator.get_confidence_score(data)
        except Exception as e:
            self.logger.error(f"予測信頼度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        現在のポジション状態を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        try:
            return self.signal_generator.get_current_position(data)
        except Exception as e:
            self.logger.error(f"現在のポジション状態取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_change_points(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        トレンド変化点を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (ロングエントリーポイント, ショートエントリーポイント)
        """
        try:
            return self.signal_generator.get_trend_change_points(data)
        except Exception as e:
            self.logger.error(f"トレンド変化点取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_current_state(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        現在の状態情報を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, Any]: 現在の状態情報
        """
        try:
            return self.signal_generator.get_current_state(data)
        except Exception as e:
            self.logger.error(f"現在の状態情報取得中にエラー: {str(e)}")
            return {}
    
    def get_ultimate_chop_trend_result(self, data: Union[pd.DataFrame, np.ndarray]) -> object:
        """
        Ultimate Chop Trend V3の完全な結果を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Ultimate Chop Trend V3の計算結果オブジェクト
        """
        try:
            return self.signal_generator.get_ultimate_chop_trend_result(data)
        except Exception as e:
            self.logger.error(f"Ultimate Chop Trend V3結果取得中にエラー: {str(e)}")
            return None
    
    def get_all_components(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Ultimate Chop Trend V3の全コンポーネントを取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, np.ndarray]: 全コンポーネントの辞書
        """
        try:
            return self.signal_generator.get_all_components(data)
        except Exception as e:
            self.logger.error(f"全コンポーネント取得中にエラー: {str(e)}")
            return {}
    
    def get_filter_stats(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        フィルタリング統計を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, Any]: フィルタリング統計
        """
        try:
            return self.signal_generator.get_filter_stats(data)
        except Exception as e:
            self.logger.error(f"フィルタリング統計取得中にエラー: {str(e)}")
            return {}
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            # Ultimate Chop Trend V3パラメータ
            'analysis_period': trial.suggest_int('analysis_period', 5, 50),
            'fast_period': trial.suggest_int('fast_period', 3, 20),
            'trend_threshold': trial.suggest_float('trend_threshold', 0.4, 0.8, step=0.02),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.1, 0.6, step=0.05),
            
            # アルゴリズム有効化（軽量なものを中心に）
            'enable_hilbert': trial.suggest_categorical('enable_hilbert', [True, False]),
            'enable_regression': trial.suggest_categorical('enable_regression', [True, False]),
            'enable_consensus': trial.suggest_categorical('enable_consensus', [True, False]),
            'enable_volatility': trial.suggest_categorical('enable_volatility', [True, False]),
            'enable_zerollag': trial.suggest_categorical('enable_zerollag', [True, False]),
            
            # シグナル生成パラメータ
            'min_confidence_for_entry': trial.suggest_float('min_confidence_for_entry', 0.2, 0.7, step=0.05),
            'require_strong_confidence': trial.suggest_categorical('require_strong_confidence', [True, False]),
            'use_strong_exit': trial.suggest_categorical('use_strong_exit', [True, False]),
            'confidence_filter_strength': trial.suggest_float('confidence_filter_strength', 0.0, 0.6, step=0.05),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        strategy_params = {
            'analysis_period': int(params.get('analysis_period', 14)),
            'fast_period': int(params.get('fast_period', 7)),
            'trend_threshold': float(params.get('trend_threshold', 0.58)),
            'confidence_threshold': float(params.get('confidence_threshold', 0.3)),
            
            # アルゴリズム有効化
            'enable_hilbert': bool(params.get('enable_hilbert', True)),
            'enable_regression': bool(params.get('enable_regression', True)),
            'enable_consensus': bool(params.get('enable_consensus', True)),
            'enable_volatility': bool(params.get('enable_volatility', True)),
            'enable_zerollag': bool(params.get('enable_zerollag', True)),
            
            # シグナル生成パラメータ
            'enable_exit_signals': True,  # 常にTrue
            'min_confidence_for_entry': float(params.get('min_confidence_for_entry', 0.4)),
            'require_strong_confidence': bool(params.get('require_strong_confidence', True)),
            'use_strong_exit': bool(params.get('use_strong_exit', False)),
            'confidence_filter_strength': float(params.get('confidence_filter_strength', 0.3)),
        }
        return strategy_params
    
    def get_parameter_ranges(self) -> Dict[str, Any]:
        """
        パラメータ範囲を取得
        
        Returns:
            Dict[str, Any]: パラメータ範囲の辞書
        """
        return {
            'analysis_period': {
                'type': 'int',
                'min': 5,
                'max': 50,
                'default': 14,
                'description': '分析期間'
            },
            'fast_period': {
                'type': 'int',
                'min': 3,
                'max': 20,
                'default': 7,
                'description': '高速期間'
            },
            'trend_threshold': {
                'type': 'float',
                'min': 0.4,
                'max': 0.8,
                'step': 0.02,
                'default': 0.58,
                'description': 'トレンド判定しきい値'
            },
            'confidence_threshold': {
                'type': 'float',
                'min': 0.1,
                'max': 0.6,
                'step': 0.05,
                'default': 0.3,
                'description': '信頼度しきい値'
            },
            'min_confidence_for_entry': {
                'type': 'float',
                'min': 0.2,
                'max': 0.7,
                'step': 0.05,
                'default': 0.4,
                'description': 'エントリー最小信頼度'
            },
            'confidence_filter_strength': {
                'type': 'float',
                'min': 0.0,
                'max': 0.6,
                'step': 0.05,
                'default': 0.3,
                'description': '信頼度フィルター強度'
            },
            'enable_hilbert': {
                'type': 'bool',
                'default': True,
                'description': 'ヒルベルト変換を有効にする'
            },
            'enable_regression': {
                'type': 'bool',
                'default': True,
                'description': '増分回帰を有効にする'
            },
            'enable_consensus': {
                'type': 'bool',
                'default': True,
                'description': 'コンセンサスを有効にする'
            },
            'enable_volatility': {
                'type': 'bool',
                'default': True,
                'description': 'ボラティリティ分析を有効にする'
            },
            'enable_zerollag': {
                'type': 'bool',
                'default': True,
                'description': 'ゼロラグEMAを有効にする'
            },
            'require_strong_confidence': {
                'type': 'bool',
                'default': True,
                'description': '強い信頼度を要求する'
            },
            'use_strong_exit': {
                'type': 'bool',
                'default': False,
                'description': '強い決済条件を使用する'
            }
        }
    
    def get_strategy_description(self) -> str:
        """
        戦略の説明を取得
        
        Returns:
            str: 戦略の説明
        """
        return """
        Ultimate Chop Trend V3ストラレジー
        
        【特徴】
        - 5つのアルゴリズム（ヒルベルト変換、増分回帰、コンセンサス、ボラティリティ分析、ゼロラグEMA）の統合
        - トレンド方向の転換点を高精度で検出
        - 信頼度による品質フィルタリング
        - Numbaによる高速化処理
        
        【エントリー条件】
        - ロング: trend_direction が -1 から 1 に変化 + 信頼度条件
        - ショート: trend_direction が 1 から -1 に変化 + 信頼度条件
        
        【エグジット条件】
        - 通常: トレンド方向が反転した時
        - 強制: トレンド方向が反転またはレンジになった時
        
        【パラメータ】
        - analysis_period: 分析期間（5-50、デフォルト14）
        - fast_period: 高速期間（3-20、デフォルト7）
        - trend_threshold: トレンド判定しきい値（0.4-0.8、デフォルト0.58）
        - min_confidence_for_entry: エントリー最小信頼度（0.2-0.7、デフォルト0.4）
        """ 