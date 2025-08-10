#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **Unified Kalman Filter - 統合カルマンフィルター** 🚀

kalmanディレクトリ内の全てのカルマンフィルターを統合し、
単一のインターフェースで利用可能にします。

🌟 **対応フィルター:**
- 'adaptive': 適応カルマンフィルター (Adaptive Kalman Filter)
- 'multivariate': 多変量カルマンフィルター (Multivariate Kalman Filter)
- 'quantum_adaptive': 量子適応カルマンフィルター (Quantum Adaptive Kalman Filter)
- 'simple': シンプルカルマンフィルター (Simple Kalman Filter - パインスクリプト互換)
- 'unscented': 無香料カルマンフィルター (Unscented Kalman Filter)
- 'unscented_v2': 無香料カルマンフィルターV2 (Unscented Kalman Filter V2)

📊 **特徴:**
- 複数のカルマンフィルターアルゴリズムを選択可能
- 統一されたインターフェースとパラメータ管理
- プライスソース対応
- キャッシュによるパフォーマンス最適化
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, Type
import numpy as np
import pandas as pd
import traceback

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
    # カルマンフィルターのインポート
    from .adaptive_kalman import AdaptiveKalman, AdaptiveKalmanResult
    from .multivariate_kalman import MultivariateKalman, MultivariateKalmanResult
    from .quantum_adaptive_kalman import QuantumAdaptiveKalman, QuantumAdaptiveKalmanResult
    from .simple_kalman import SimpleKalman, SimpleKalmanResult
    from .unscented_kalman_filter import UnscentedKalmanFilter, UnscentedKalmanResult
    from .unscented_kalman_filter_v2 import UnscentedKalmanFilterV2Wrapper, UKFResult
except ImportError:
    # Fallback for potential execution context issues
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator import Indicator
    from price_source import PriceSource
    # カルマンフィルターのインポート (絶対パス)
    from indicators.kalman.adaptive_kalman import AdaptiveKalman, AdaptiveKalmanResult
    from indicators.kalman.multivariate_kalman import MultivariateKalman, MultivariateKalmanResult
    from indicators.kalman.quantum_adaptive_kalman import QuantumAdaptiveKalman, QuantumAdaptiveKalmanResult
    from indicators.kalman.simple_kalman import SimpleKalman, SimpleKalmanResult
    from indicators.kalman.unscented_kalman_filter import UnscentedKalmanFilter, UnscentedKalmanResult
    from indicators.kalman.unscented_kalman_filter_v2 import UnscentedKalmanFilterV2Wrapper, UKFResult


@dataclass
class UnifiedKalmanResult:
    """統合カルマンフィルターの計算結果"""
    values: np.ndarray                    # フィルタリングされた値（メイン結果）
    filter_type: str                      # 使用されたフィルタータイプ
    parameters: Dict[str, Any]            # 使用されたパラメータ
    additional_data: Dict[str, np.ndarray]  # フィルター固有の追加データ
    raw_values: np.ndarray                # 元の価格データ


class UnifiedKalman(Indicator):
    """
    統合カルマンフィルター
    
    kalmanディレクトリ内の全てのカルマンフィルターを統一インターフェースで利用：
    - 複数のカルマンフィルターアルゴリズムを選択可能
    - 一貫したパラメータ設定
    - プライスソース対応
    - キャッシュによるパフォーマンス最適化
    """
    
    # 利用可能なフィルターの定義
    _FILTERS = {
        'adaptive': AdaptiveKalman,
        'multivariate': MultivariateKalman,
        'quantum_adaptive': QuantumAdaptiveKalman,
        'simple': SimpleKalman,
        'unscented': UnscentedKalmanFilter,
        'unscented_v2': UnscentedKalmanFilterV2Wrapper,
    }
    
    # フィルターの説明
    _FILTER_DESCRIPTIONS = {
        'adaptive': '適応カルマンフィルター（Adaptive Kalman Filter）',
        'multivariate': '多変量カルマンフィルター（Multivariate Kalman Filter）',
        'quantum_adaptive': '量子適応カルマンフィルター（Quantum Adaptive Kalman Filter）',
        'simple': 'シンプルカルマンフィルター（Simple Kalman Filter - パインスクリプト互換）',
        'unscented': '無香料カルマンフィルター（Unscented Kalman Filter）',
        'unscented_v2': '無香料カルマンフィルターV2（Unscented Kalman Filter V2）',
    }
    
    # デフォルトパラメータ
    _DEFAULT_PARAMS = {
        'adaptive': {
            'process_noise': 1e-5,
            'min_observation_noise': 1e-6,
            'adaptation_window': 5
        },
        'multivariate': {
            'process_noise': 1e-5,
            'observation_noise': 1e-3,
            'volatility_noise': 1e-4
        },
        'quantum_adaptive': {
            'base_process_noise': 0.001,
            'amplitude_window': 14,
            'coherence_lookback': 5
        },
        'simple': {
            'R': 0.1,
            'Q': 0.01,
            'initial_covariance': 1.0,
            'enable_trend_detection': True
        },
        'unscented': {
            'alpha': 0.1,
            'beta': 2.0,
            'kappa': 0.0,
            'process_noise_scale': 0.01,
            'volatility_window': 10,
            'adaptive_noise': True
        },
        'unscented_v2': {
            'kappa': 0.0,
            'process_noise_scale': 0.01,
            'observation_noise_scale': 0.001,
            'max_steps': 1000
        }
    }
    
    def __init__(
        self,
        filter_type: str = 'adaptive',
        src_type: str = 'close',
        **kwargs
    ):
        """
        コンストラクタ
        
        Args:
            filter_type: 使用するフィルタータイプ
                - 'adaptive': 適応カルマンフィルター
                - 'multivariate': 多変量カルマンフィルター
                - 'quantum_adaptive': 量子適応カルマンフィルター
                - 'simple': シンプルカルマンフィルター（パインスクリプト互換）
                - 'unscented': 無香料カルマンフィルター
                - 'unscented_v2': 無香料カルマンフィルターV2
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            **kwargs: フィルター固有のパラメータ
        """
        # フィルタータイプの正規化
        filter_type = filter_type.lower()
        
        # フィルタータイプの検証
        if filter_type not in self._FILTERS:
            raise ValueError(
                f"無効なフィルタータイプ: {filter_type}。"
                f"有効なオプション: {', '.join(self._FILTERS.keys())}"
            )
        
        # インディケーター名の設定
        indicator_name = f"UnifiedKalman({filter_type}, src={src_type})"
        super().__init__(indicator_name)
        
        self.filter_type = filter_type
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open', 'oc2']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
        # パラメータの設定
        self.parameters = self._DEFAULT_PARAMS[filter_type].copy()
        self.parameters.update(kwargs)
        
        # フィルターインスタンスの作成
        self.filter = self._create_filter_instance()
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _create_filter_instance(self):
        """フィルターインスタンスを作成"""
        filter_class = self._FILTERS[self.filter_type]
        
        try:
            # 各フィルターのコンストラクタに応じてパラメータを調整
            if self.filter_type == 'adaptive':
                return filter_class(
                    process_noise=self.parameters.get('process_noise', 1e-5),
                    src_type=self.src_type,
                    min_observation_noise=self.parameters.get('min_observation_noise', 1e-6),
                    adaptation_window=self.parameters.get('adaptation_window', 5)
                )
            
            elif self.filter_type == 'multivariate':
                return filter_class(
                    process_noise=self.parameters.get('process_noise', 1e-5),
                    observation_noise=self.parameters.get('observation_noise', 1e-3),
                    volatility_noise=self.parameters.get('volatility_noise', 1e-4)
                )
            
            elif self.filter_type == 'quantum_adaptive':
                return filter_class(
                    src_type=self.src_type,
                    base_process_noise=self.parameters.get('base_process_noise', 0.001),
                    amplitude_window=self.parameters.get('amplitude_window', 14),
                    coherence_lookback=self.parameters.get('coherence_lookback', 5)
                )
            
            elif self.filter_type == 'simple':
                return filter_class(
                    R=self.parameters.get('R', 0.1),
                    Q=self.parameters.get('Q', 0.01),
                    src_type=self.src_type,
                    initial_covariance=self.parameters.get('initial_covariance', 1.0),
                    enable_trend_detection=self.parameters.get('enable_trend_detection', True)
                )
            
            elif self.filter_type == 'unscented':
                return filter_class(
                    src_type=self.src_type,
                    alpha=self.parameters.get('alpha', 0.1),
                    beta=self.parameters.get('beta', 2.0),
                    kappa=self.parameters.get('kappa', 0.0),
                    process_noise_scale=self.parameters.get('process_noise_scale', 0.01),
                    volatility_window=self.parameters.get('volatility_window', 10),
                    adaptive_noise=self.parameters.get('adaptive_noise', True)
                )
            
            elif self.filter_type == 'unscented_v2':
                return filter_class(
                    src_type=self.src_type,
                    kappa=self.parameters.get('kappa', 0.0),
                    process_noise_scale=self.parameters.get('process_noise_scale', 0.01),
                    observation_noise_scale=self.parameters.get('observation_noise_scale', 0.001),
                    max_steps=self.parameters.get('max_steps', 1000)
                )
            
            else:
                # 汎用コンストラクタ
                return filter_class(src_type=self.src_type, **self.parameters)
                
        except Exception as e:
            self.logger.error(f"フィルターインスタンス作成エラー ({self.filter_type}): {e}")
            raise
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                else:
                    first_val = last_val = 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            params_sig = f"{self.filter_type}_{self.src_type}_{hash(str(sorted(self.parameters.items())))}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.filter_type}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UnifiedKalmanResult:
        """
        統合カルマンフィルターを使用してフィルタリングを計算
        
        Args:
            data: 価格データ
            
        Returns:
            UnifiedKalmanResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UnifiedKalmanResult(
                    values=cached_result.values.copy(),
                    filter_type=cached_result.filter_type,
                    parameters=cached_result.parameters.copy(),
                    additional_data={k: v.copy() if isinstance(v, np.ndarray) else v 
                                   for k, v in cached_result.additional_data.items()},
                    raw_values=cached_result.raw_values.copy()
                )
            
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < 2:
                return self._create_empty_result(data_length, src_prices)
            
            # フィルターの計算実行
            filter_result = self.filter.calculate(data)
            
            # 結果の標準化（各フィルター固有の処理）
            filtered_values, additional_data = self._standardize_result(filter_result)
            
            # NumPy配列への変換（必要に応じて）
            if not isinstance(filtered_values, np.ndarray):
                filtered_values = np.array(filtered_values)
            
            # 結果の作成
            result = UnifiedKalmanResult(
                values=filtered_values.copy(),
                filter_type=self.filter_type,
                parameters=self.parameters.copy(),
                additional_data=additional_data,
                raw_values=src_prices.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = filtered_values
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"統合カルマンフィルター計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _standardize_result(self, filter_result) -> tuple:
        """
        フィルター結果を標準化して、統一された形式で返す
        
        Args:
            filter_result: 各フィルターの計算結果
            
        Returns:
            tuple: (filtered_values, additional_data)
        """
        additional_data = {}
        
        try:
            # 各フィルター固有の結果処理
            if self.filter_type == 'simple':
                # SimpleKalmanResult
                filtered_values = filter_result.values
                additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
                additional_data['error_covariances'] = filter_result.error_covariances.copy()
                additional_data['predictions'] = filter_result.predictions.copy()
                additional_data['trend_signals'] = filter_result.trend_signals.copy()
                
            elif self.filter_type == 'unscented':
                # UKFResult
                filtered_values = filter_result.filtered_values
                additional_data['velocity_estimates'] = filter_result.velocity_estimates.copy()
                additional_data['acceleration_estimates'] = filter_result.acceleration_estimates.copy()
                additional_data['uncertainty'] = filter_result.uncertainty.copy()
                additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
                additional_data['innovations'] = filter_result.innovations.copy()
                additional_data['confidence_scores'] = filter_result.confidence_scores.copy()
                
            elif self.filter_type == 'unscented_v2':
                # UKFResult from V2
                if hasattr(filter_result, 'filtered_values'):
                    filtered_values = filter_result.filtered_values
                elif hasattr(filter_result, 'values'):
                    filtered_values = filter_result.values
                else:
                    filtered_values = filter_result
                self._extract_ukf_additional_data(filter_result, additional_data)
                
            elif self.filter_type == 'quantum_adaptive':
                # QuantumAdaptiveKalmanResult
                filtered_values = filter_result.values
                additional_data['quantum_coherence'] = filter_result.quantum_coherence.copy()
                additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
                additional_data['innovations'] = filter_result.innovations.copy()
                additional_data['confidence_scores'] = filter_result.confidence_scores.copy()
                
            elif self.filter_type == 'adaptive':
                # AdaptiveKalmanResult
                filtered_values = filter_result.filtered_signal
                if hasattr(filter_result, 'adaptive_gain'):
                    additional_data['adaptive_gain'] = filter_result.adaptive_gain.copy()
                if hasattr(filter_result, 'error_covariance'):
                    additional_data['error_covariance'] = filter_result.error_covariance.copy()
                if hasattr(filter_result, 'innovations'):
                    additional_data['innovations'] = filter_result.innovations.copy()
                if hasattr(filter_result, 'confidence_score'):
                    additional_data['confidence_score'] = filter_result.confidence_score.copy()
                
            elif self.filter_type == 'multivariate':
                # MultivariateKalmanResult
                filtered_values = filter_result.filtered_prices
                if hasattr(filter_result, 'price_range_estimates'):
                    additional_data['price_range_estimates'] = filter_result.price_range_estimates.copy()
                if hasattr(filter_result, 'volatility_estimates'):
                    additional_data['volatility_estimates'] = filter_result.volatility_estimates.copy()
                if hasattr(filter_result, 'state_estimates'):
                    additional_data['state_estimates'] = filter_result.state_estimates.copy()
                if hasattr(filter_result, 'velocity_estimates'):
                    additional_data['velocity_estimates'] = filter_result.velocity_estimates.copy()
                if hasattr(filter_result, 'filtered_high'):
                    additional_data['filtered_high'] = filter_result.filtered_high.copy()
                if hasattr(filter_result, 'filtered_low'):
                    additional_data['filtered_low'] = filter_result.filtered_low.copy()
                if hasattr(filter_result, 'filtered_close'):
                    additional_data['filtered_close'] = filter_result.filtered_close.copy()
                if hasattr(filter_result, 'kalman_gains'):
                    additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
                if hasattr(filter_result, 'innovations'):
                    additional_data['innovations'] = filter_result.innovations.copy()
                if hasattr(filter_result, 'confidence_scores'):
                    additional_data['confidence_scores'] = filter_result.confidence_scores.copy()
                
            else:
                # 汎用処理
                if hasattr(filter_result, 'values'):
                    filtered_values = filter_result.values
                elif hasattr(filter_result, 'filtered_values'):
                    filtered_values = filter_result.filtered_values
                elif hasattr(filter_result, 'filtered_signal'):
                    filtered_values = filter_result.filtered_signal
                elif hasattr(filter_result, 'filtered_prices'):
                    filtered_values = filter_result.filtered_prices
                else:
                    filtered_values = filter_result
                    
                # 共通の追加データを抽出
                self._extract_common_additional_data(filter_result, additional_data)
                    
        except Exception as e:
            self.logger.error(f"結果標準化エラー ({self.filter_type}): {e}")
            # エラー時はフィルター結果をそのまま使用
            if hasattr(filter_result, 'values'):
                filtered_values = filter_result.values
            elif hasattr(filter_result, 'filtered_values'):
                filtered_values = filter_result.filtered_values
            elif hasattr(filter_result, 'filtered_signal'):
                filtered_values = filter_result.filtered_signal
            elif hasattr(filter_result, 'filtered_prices'):
                filtered_values = filter_result.filtered_prices
            else:
                filtered_values = filter_result
                
        return filtered_values, additional_data
    
    def _extract_ukf_additional_data(self, filter_result, additional_data: Dict[str, np.ndarray]):
        """UKF固有の追加データを抽出"""
        try:
            if hasattr(filter_result, 'velocity_estimates'):
                additional_data['velocity_estimates'] = filter_result.velocity_estimates.copy()
            if hasattr(filter_result, 'acceleration_estimates'):
                additional_data['acceleration_estimates'] = filter_result.acceleration_estimates.copy()
            if hasattr(filter_result, 'uncertainty'):
                additional_data['uncertainty'] = filter_result.uncertainty.copy()
            if hasattr(filter_result, 'kalman_gains'):
                additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
            if hasattr(filter_result, 'innovations'):
                additional_data['innovations'] = filter_result.innovations.copy()
            if hasattr(filter_result, 'confidence_scores'):
                additional_data['confidence_scores'] = filter_result.confidence_scores.copy()
        except Exception as e:
            self.logger.warning(f"UKF追加データ抽出エラー: {e}")
            
    def _extract_common_additional_data(self, filter_result, additional_data: Dict[str, np.ndarray]):
        """共通の追加データを抽出"""
        try:
            common_attributes = [
                'kalman_gains', 'innovations', 'error_covariance', 'confidence_score',
                'confidence_scores', 'uncertainty', 'predictions', 'trend_signals'
            ]
            
            for attr in common_attributes:
                if hasattr(filter_result, attr):
                    value = getattr(filter_result, attr)
                    if isinstance(value, np.ndarray):
                        additional_data[attr] = value.copy()
                    else:
                        additional_data[attr] = np.array(value) if value is not None else np.array([])
        except Exception as e:
            self.logger.warning(f"共通追加データ抽出エラー: {e}")
            
    def _extract_additional_data(self, filter_result) -> Dict[str, np.ndarray]:
        """フィルター固有の追加データを抽出（後方互換性のため残す）"""
        additional_data = {}
        
        try:
            # 適応カルマンフィルター固有データ
            if hasattr(filter_result, 'adaptive_gain'):
                additional_data['adaptive_gain'] = filter_result.adaptive_gain.copy()
            if hasattr(filter_result, 'innovations'):
                additional_data['innovations'] = filter_result.innovations.copy()
            if hasattr(filter_result, 'error_covariance'):
                additional_data['error_covariance'] = filter_result.error_covariance.copy()
            if hasattr(filter_result, 'confidence_score'):
                additional_data['confidence_score'] = filter_result.confidence_score.copy()
            
            # 多変量カルマンフィルター固有データ
            if hasattr(filter_result, 'price_range_estimates'):
                additional_data['price_range_estimates'] = filter_result.price_range_estimates.copy()
            if hasattr(filter_result, 'volatility_estimates'):
                additional_data['volatility_estimates'] = filter_result.volatility_estimates.copy()
            if hasattr(filter_result, 'state_estimates'):
                additional_data['state_estimates'] = filter_result.state_estimates.copy()
            if hasattr(filter_result, 'uncertainty_estimates'):
                additional_data['uncertainty_estimates'] = filter_result.uncertainty_estimates.copy()
            
            # 量子適応カルマンフィルター固有データ
            if hasattr(filter_result, 'quantum_coherence'):
                additional_data['quantum_coherence'] = filter_result.quantum_coherence.copy()
            if hasattr(filter_result, 'kalman_gains'):
                additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
            if hasattr(filter_result, 'confidence_scores'):
                additional_data['confidence_scores'] = filter_result.confidence_scores.copy()
            
            # シンプルカルマンフィルター固有データ
            if hasattr(filter_result, 'kalman_gains'):
                additional_data['kalman_gains'] = filter_result.kalman_gains.copy()
            if hasattr(filter_result, 'error_covariances'):
                additional_data['error_covariances'] = filter_result.error_covariances.copy()
            if hasattr(filter_result, 'predictions'):
                additional_data['predictions'] = filter_result.predictions.copy()
            if hasattr(filter_result, 'trend_signals'):
                additional_data['trend_signals'] = filter_result.trend_signals.copy()
            
            # 無香料カルマンフィルター固有データ
            if hasattr(filter_result, 'velocity_estimates'):
                additional_data['velocity_estimates'] = filter_result.velocity_estimates.copy()
            if hasattr(filter_result, 'acceleration_estimates'):
                additional_data['acceleration_estimates'] = filter_result.acceleration_estimates.copy()
            if hasattr(filter_result, 'uncertainty'):
                additional_data['uncertainty'] = filter_result.uncertainty.copy()
            if hasattr(filter_result, 'sigma_points'):
                additional_data['sigma_points'] = filter_result.sigma_points.copy()
            
        except Exception as e:
            self.logger.warning(f"追加データ抽出中にエラー: {e}")
        
        return additional_data
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> UnifiedKalmanResult:
        """空の結果を作成"""
        return UnifiedKalmanResult(
            values=np.full(length, np.nan),
            filter_type=self.filter_type,
            parameters=self.parameters.copy(),
            additional_data={},
            raw_values=raw_prices
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルタリングされた値を取得"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_raw_values(self) -> Optional[np.ndarray]:
        """元の価格データを取得"""
        result = self._get_latest_result()
        return result.raw_values.copy() if result else None
    
    def get_additional_data(self, key: str) -> Optional[np.ndarray]:
        """フィルター固有の追加データを取得"""
        result = self._get_latest_result()
        if result and key in result.additional_data:
            return result.additional_data[key].copy()
        return None
    
    def get_filter_info(self) -> Dict[str, Any]:
        """フィルター情報を取得"""
        return {
            'type': self.filter_type,
            'description': self._FILTER_DESCRIPTIONS.get(self.filter_type, 'Unknown'),
            'src_type': self.src_type,
            'parameters': self.parameters.copy()
        }
    
    def _get_latest_result(self) -> Optional[UnifiedKalmanResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        if hasattr(self.filter, 'reset'):
            self.filter.reset()
        self._result_cache = {}
        self._cache_keys = []
    
    @classmethod
    def get_available_filters(cls) -> Dict[str, str]:
        """
        利用可能なフィルターとその説明を返す
        
        Returns:
            Dict[str, str]: フィルター名とその説明の辞書
        """
        return cls._FILTER_DESCRIPTIONS.copy()
    
    @classmethod
    def get_default_parameters(cls, filter_type: str) -> Dict[str, Any]:
        """
        指定されたフィルターのデフォルトパラメータを取得
        
        Args:
            filter_type: フィルタータイプ
            
        Returns:
            Dict[str, Any]: デフォルトパラメータ
        """
        filter_type = filter_type.lower()
        return cls._DEFAULT_PARAMS.get(filter_type, {}).copy()


# 便利関数
def filter_data(
    data: Union[pd.DataFrame, np.ndarray], 
    filter_type: str = 'adaptive',
    src_type: str = 'close',
    **kwargs
) -> np.ndarray:
    """
    統合カルマンフィルターの計算（便利関数）
    
    Args:
        data: 価格データ
        filter_type: フィルタータイプ
        src_type: 価格ソース
        **kwargs: フィルター固有のパラメータ
        
    Returns:
        フィルタリングされた値
    """
    kalman_filter = UnifiedKalman(filter_type=filter_type, src_type=src_type, **kwargs)
    result = kalman_filter.calculate(data)
    return result.values


def compare_filters(
    data: Union[pd.DataFrame, np.ndarray],
    filter_types: list = ['adaptive', 'quantum_adaptive', 'unscented'],
    src_type: str = 'close',
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    複数のカルマンフィルターを比較
    
    Args:
        data: 価格データ
        filter_types: 比較するフィルタータイプのリスト
        src_type: 価格ソース
        **kwargs: フィルター固有のパラメータ
        
    Returns:
        Dict[str, np.ndarray]: フィルタータイプ別の結果
    """
    results = {}
    
    for filter_type in filter_types:
        try:
            kalman_filter = UnifiedKalman(filter_type=filter_type, src_type=src_type, **kwargs)
            result = kalman_filter.calculate(data)
            results[filter_type] = result.values
        except Exception as e:
            print(f"Error with {filter_type}: {e}")
            results[filter_type] = np.full(len(data), np.nan)
    
    return results


# テスト実行部分
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== 統合カルマンフィルターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    trend = 0.001
    volatility = 0.02
    
    prices = [base_price]
    for i in range(1, length):
        change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, volatility * close * 0.5))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * close * 0.2)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 利用可能なフィルターを表示
    filters = UnifiedKalman.get_available_filters()
    print(f"\n利用可能なフィルター: {len(filters)}種類")
    for name, desc in filters.items():
        print(f"  {name}: {desc}")
    
    # 各フィルターをテスト
    test_filters = ['adaptive', 'quantum_adaptive', 'unscented']
    print(f"\nテスト対象: {test_filters}")
    
    results = {}
    for filter_type in test_filters:
        try:
            print(f"\n{filter_type} をテスト中...")
            kalman_filter = UnifiedKalman(filter_type=filter_type, src_type='close')
            result = kalman_filter.calculate(df)
            
            mean_filtered = np.nanmean(result.values)
            mean_raw = np.nanmean(result.raw_values)
            valid_count = np.sum(~np.isnan(result.values))
            
            results[filter_type] = result
            
            print(f"  平均値: {mean_filtered:.4f} (元: {mean_raw:.4f})")
            print(f"  有効値数: {valid_count}/{len(df)}")
            print(f"  追加データ: {list(result.additional_data.keys()) if result.additional_data else 'なし'}")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n=== テスト完了 ===")