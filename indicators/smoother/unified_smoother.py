#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Unified Smoother - 統合スムーサー** 🎯

スムーサーディレクトリ内の全スムーサーを統合した統一インターフェース：
- 複数のスムージング手法を選択可能
- 一貫したインターフェースで利用
- パフォーマンス最適化とキャッシュ機能
- プライスソース対応

🌟 **対応スムーサー:**
1. **FRAMA**: Fractal Adaptive Moving Average
2. **Super Smoother**: エーラーズ・スーパースムーサー
3. **Ultimate Smoother**: 究極スムーサー  
4. **Zero Lag EMA**: ゼロラグ指数移動平均

📊 **使用例:**
```python
smoother = UnifiedSmoother(smoother_type='frama', period=21)
result = smoother.calculate(data)
```
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
import traceback

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
    # スムーサーのインポート
    from .frama import FRAMA
    from .super_smoother import SuperSmoother
    from .ultimate_smoother import UltimateSmoother
    from .zero_lag_ema import ZeroLagEMA
    from .laguerre_filter import LaguerreFilter
    from .alma import ALMA
    from .hma import HMA
    # カルマンフィルターのインポート
    from ..kalman.unified_kalman import UnifiedKalman
    
    # Ultimate MA をインポート
    try:
        from .ultimate_ma import UltimateMA
    except ImportError:
        UltimateMA = None
        print("Warning: Ultimate MA のインポートに失敗しました")
        
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    # プロジェクトルートをパスに追加
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource
    # スムーサーのインポート
    from indicators.smoother.frama import FRAMA
    from indicators.smoother.super_smoother import SuperSmoother
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    from indicators.smoother.zero_lag_ema import ZeroLagEMA
    from indicators.smoother.laguerre_filter import LaguerreFilter
    from indicators.smoother.alma import ALMA
    from indicators.smoother.hma import HMA
    # カルマンフィルターのインポート
    from indicators.kalman.unified_kalman import UnifiedKalman
    
    # Ultimate MA をインポート
    try:
        from indicators.smoother.ultimate_ma import UltimateMA
    except ImportError:
        UltimateMA = None
        print("Warning: Ultimate MA のインポートに失敗しました")


@dataclass
class UnifiedSmootherResult:
    """統合スムーサーの計算結果"""
    values: np.ndarray           # スムースされた値
    raw_values: np.ndarray       # 元の価格データ
    kalman_filtered_values: Optional[np.ndarray]  # カルマンフィルター後の値（使用時のみ）
    smoother_type: str           # 使用されたスムーサータイプ
    kalman_type: Optional[str]   # 使用されたカルマンタイプ（使用時のみ）
    parameters: Dict[str, Any]   # 使用されたパラメータ
    kalman_parameters: Dict[str, Any]  # カルマンパラメータ（使用時のみ）
    additional_data: Dict[str, np.ndarray]  # スムーサー固有の追加データ


class UnifiedSmoother(Indicator):
    """
    統合スムーサー
    
    スムーサーディレクトリ内の全スムーサーを統一インターフェースで利用：
    - 複数のスムージング手法を選択可能
    - 一貫したパラメータ設定
    - プライスソース対応
    - キャッシュによるパフォーマンス最適化
    """
    
    # 利用可能なスムーザーの定義
    _SMOOTHERS = {
        'frama': FRAMA,
        'super_smoother': SuperSmoother,
        'ultimate_smoother': UltimateSmoother,
        'zero_lag_ema': ZeroLagEMA,
        'zlema': ZeroLagEMA,  # エイリアス
        'laguerre_filter': LaguerreFilter,
        'laguerre': LaguerreFilter,  # エイリアス
        'alma': ALMA,
        'hma': HMA,
    }
    
    # 条件付きで Ultimate MA を追加
    if UltimateMA is not None:
        _SMOOTHERS['ultimate_ma'] = UltimateMA
    
    # スムーサーの説明
    _SMOOTHER_DESCRIPTIONS = {
        'frama': 'FRAMA（フラクタル適応移動平均）',
        'super_smoother': 'スーパースムーサー（エーラーズ2極フィルター）',
        'ultimate_smoother': '究極スムーサー（高度適応フィルター）',
        'zero_lag_ema': 'ゼロラグEMA（遅延除去指数移動平均）',
        'zlema': 'ゼロラグEMA（遅延除去指数移動平均）',
        'laguerre_filter': 'ラゲールフィルター（時間軸歪みスムーサー）',
        'laguerre': 'ラゲールフィルター（時間軸歪みスムーサー）',
        'alma': 'ALMA（Arnaud Legoux移動平均）',
        'hma': 'HMA（Hull移動平均）',
    }
    
    # 条件付きで Ultimate MA の説明を追加
    if UltimateMA is not None:
        _SMOOTHER_DESCRIPTIONS['ultimate_ma'] = 'Ultimate MA（6段階革新的フィルタリングシステム）'
    
    # デフォルトパラメータ
    _DEFAULT_PARAMS = {
        'frama': {'period': 10, 'fc': 1, 'sc': 198},
        'super_smoother': {'length': 5, 'num_poles': 2},
        'ultimate_smoother': {'period': 20.0, 'src_type': 'close'},
        'zero_lag_ema': {'period': 21, 'fast_mode': False},
        'zlema': {'period': 21, 'fast_mode': False},
        'laguerre_filter': {'gamma': 0.5, 'order': 4, 'period': 4},
        'laguerre': {'gamma': 0.8, 'order': 4, 'period': 4},
        'alma': {'length': 9, 'offset': 0.85, 'sigma': 6.0},
        'hma': {'length': 14},
    }
    
    # 条件付きで Ultimate MA のデフォルトパラメータを追加
    if UltimateMA is not None:
        _DEFAULT_PARAMS['ultimate_ma'] = {
            'ultimate_smoother_period': 5.0,
            'zero_lag_period': 21,
            'realtime_window': 89,
            'src_type': 'hlc3',
            'slope_index': 1,
            'range_threshold': 0.005,
            'use_adaptive_kalman': True,
            'zero_lag_period_mode': 'fixed',
            'realtime_window_mode': 'fixed'
        }
    
    def __init__(
        self,
        smoother_type: str = 'frama',
        src_type: str = 'close',
        period_mode: str = 'fixed',
        enable_kalman: bool = False,
        kalman_type: str = 'simple',
        **kwargs
    ):
        """
        コンストラクタ
        
        Args:
            smoother_type: 使用するスムーサータイプ
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            period_mode: 期間モード ('fixed' または 'dynamic', ultimate_smootherのみ対応)
            enable_kalman: カルマンフィルターを有効にするか
            kalman_type: カルマンフィルタータイプ ('simple', 'adaptive', 'quantum_adaptive', etc.)
            **kwargs: スムーサー・カルマンフィルター固有のパラメータ
        """
        # スムーサータイプの正規化
        smoother_type = smoother_type.lower()
        
        # スムーサータイプの検証
        if smoother_type not in self._SMOOTHERS:
            raise ValueError(
                f"無効なスムーサータイプ: {smoother_type}。"
                f"有効なオプション: {', '.join(self._SMOOTHERS.keys())}"
            )
        
        # インディケーター名の設定
        kalman_suffix = f"+{kalman_type}" if enable_kalman else ""
        indicator_name = f"UnifiedSmoother({smoother_type}{kalman_suffix}, src={src_type})"
        super().__init__(indicator_name)
        
        self.smoother_type = smoother_type
        self.src_type = src_type.lower()
        self.period_mode = period_mode.lower()
        self.enable_kalman = enable_kalman
        self.kalman_type = kalman_type.lower() if enable_kalman else None
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
        # 期間モードの検証
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効な期間モード: {period_mode}")
        
        # 動的期間対応スムーサーのチェック
        dynamic_supported_smoothers = ['ultimate_smoother', 'frama', 'super_smoother', 'zero_lag_ema', 'zlema', 'laguerre_filter', 'laguerre', 'alma', 'hma']
        if self.period_mode == 'dynamic' and smoother_type not in dynamic_supported_smoothers:
            self.logger.warning(
                f"{smoother_type}は動的期間に対応していません。固定期間モードに変更します。"
            )
            self.period_mode = 'fixed'
        
        # パラメータの分離
        smoother_params = {}
        kalman_params = {}
        
        for key, value in kwargs.items():
            if key.startswith('kalman_'):
                kalman_params[key[7:]] = value  # 'kalman_' プレフィックスを除去
            else:
                smoother_params[key] = value
        
        # スムーサーパラメータの設定
        self.parameters = self._DEFAULT_PARAMS[smoother_type].copy()
        self.parameters.update(smoother_params)
        
        # カルマンパラメータの設定
        self.kalman_parameters = kalman_params
        
        # 動的期間パラメータの追加（ultimate_smootherのみ）
        if self.period_mode == 'dynamic' and smoother_type == 'ultimate_smoother':
            self.parameters['period_mode'] = 'dynamic'
        
        # カルマンフィルターインスタンスの作成（必要時）
        self.kalman_filter = None
        if self.enable_kalman:
            self.kalman_filter = self._create_kalman_instance()
        
        # スムーサーインスタンスの作成
        self.smoother = self._create_smoother_instance()
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _create_kalman_instance(self):
        """カルマンフィルターインスタンスを作成"""
        try:
            return UnifiedKalman(
                filter_type=self.kalman_type,
                src_type=self.src_type,
                **self.kalman_parameters
            )
        except Exception as e:
            self.logger.error(f"カルマンフィルターインスタンス作成エラー ({self.kalman_type}): {e}")
            raise
    
    def _create_smoother_instance(self):
        """スムーサーインスタンスを作成"""
        smoother_class = self._SMOOTHERS[self.smoother_type]
        
        try:
            # 各スムーサーのコンストラクタに応じてパラメータを調整
            if self.smoother_type == 'frama':
                if self.period_mode == 'dynamic':
                    return smoother_class(
                        period=self.parameters.get('period', 16),
                        fc=self.parameters.get('fc', 1),
                        sc=self.parameters.get('sc', 300),
                        src_type=self.src_type,
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'hody_e'),
                        cycle_part=self.parameters.get('cycle_part', 0.5),
                        max_cycle=self.parameters.get('max_cycle', 124),
                        min_cycle=self.parameters.get('min_cycle', 13),
                        max_output=self.parameters.get('max_output', 124),
                        min_output=self.parameters.get('min_output', 13),
                        lp_period=self.parameters.get('lp_period', 13),
                        hp_period=self.parameters.get('hp_period', 124)
                    )
                else:
                    return smoother_class(
                        period=self.parameters.get('period', 16),
                        fc=self.parameters.get('fc', 1),
                        sc=self.parameters.get('sc', 300),
                        src_type=self.src_type
                    )
            
            elif self.smoother_type == 'super_smoother':
                if self.period_mode == 'dynamic':
                    return smoother_class(
                        length=self.parameters.get('length', 14),
                        num_poles=self.parameters.get('num_poles', 2),
                        src_type=self.src_type,
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'hody_e'),
                        cycle_part=self.parameters.get('cycle_part', 0.5),
                        max_cycle=self.parameters.get('max_cycle', 124),
                        min_cycle=self.parameters.get('min_cycle', 13),
                        max_output=self.parameters.get('max_output', 124),
                        min_output=self.parameters.get('min_output', 13),
                        lp_period=self.parameters.get('lp_period', 13),
                        hp_period=self.parameters.get('hp_period', 124)
                    )
                else:
                    return smoother_class(
                        length=self.parameters.get('length', 14),
                        num_poles=self.parameters.get('num_poles', 2),
                        src_type=self.src_type
                    )
            
            elif self.smoother_type == 'ultimate_smoother':
                # UltimateSmootherは動的期間に対応
                if self.period_mode == 'dynamic':
                    return smoother_class(
                        period=self.parameters.get('period', 20.0),
                        src_type=self.src_type,
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'absolute_ultimate'),
                        cycle_detector_cycle_part=self.parameters.get('cycle_part', 0.5),
                        cycle_detector_max_cycle=self.parameters.get('max_cycle', 120),
                        cycle_detector_min_cycle=self.parameters.get('min_cycle', 5),
                        cycle_period_multiplier=self.parameters.get('cycle_period_multiplier', 1.0),
                        cycle_detector_period_range=(
                            self.parameters.get('min_output', 5), 
                            self.parameters.get('max_output', 120)
                        )
                    )
                else:
                    return smoother_class(
                        period=self.parameters.get('period', 20.0),
                        src_type=self.src_type
                    )
            
            elif self.smoother_type in ['zero_lag_ema', 'zlema']:
                if self.period_mode == 'dynamic':
                    return ZeroLagEMA(
                        period=self.parameters.get('period', 21),
                        src_type=self.src_type,
                        fast_mode=self.parameters.get('fast_mode', False),
                        custom_alpha=self.parameters.get('custom_alpha', None),
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'hody_e'),
                        cycle_part=self.parameters.get('cycle_part', 0.5),
                        max_cycle=self.parameters.get('max_cycle', 124),
                        min_cycle=self.parameters.get('min_cycle', 13),
                        max_output=self.parameters.get('max_output', 124),
                        min_output=self.parameters.get('min_output', 13),
                        lp_period=self.parameters.get('lp_period', 13),
                        hp_period=self.parameters.get('hp_period', 124)
                    )
                else:
                    return ZeroLagEMA(
                        period=self.parameters.get('period', 21),
                        src_type=self.src_type,
                        fast_mode=self.parameters.get('fast_mode', False),
                        custom_alpha=self.parameters.get('custom_alpha', None)
                    )
            
            elif self.smoother_type in ['laguerre_filter', 'laguerre']:
                # ラゲールフィルター
                if self.period_mode == 'dynamic':
                    return LaguerreFilter(
                        gamma=self.parameters.get('gamma', 0.5),
                        order=self.parameters.get('order', 4),
                        coefficients=self.parameters.get('coefficients', None),
                        src_type=self.src_type,
                        period=self.parameters.get('period', 4),
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'hody_e'),
                        cycle_part=self.parameters.get('cycle_part', 0.5),
                        max_cycle=self.parameters.get('max_cycle', 124),
                        min_cycle=self.parameters.get('min_cycle', 13),
                        max_output=self.parameters.get('max_output', 124),
                        min_output=self.parameters.get('min_output', 13),
                        lp_period=self.parameters.get('lp_period', 13),
                        hp_period=self.parameters.get('hp_period', 124)
                    )
                else:
                    return LaguerreFilter(
                        gamma=self.parameters.get('gamma', 0.5),
                        order=self.parameters.get('order', 4),
                        coefficients=self.parameters.get('coefficients', None),
                        src_type=self.src_type,
                        period=self.parameters.get('period', 4)
                    )
            
            elif self.smoother_type == 'alma':
                # ALMA - Arnaud Legoux Moving Average
                if self.period_mode == 'dynamic':
                    return ALMA(
                        length=self.parameters.get('length', 9),
                        offset=self.parameters.get('offset', 0.85),
                        sigma=self.parameters.get('sigma', 6.0),
                        src_type=self.src_type,
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'hody_e'),
                        cycle_part=self.parameters.get('cycle_part', 0.5),
                        max_cycle=self.parameters.get('max_cycle', 124),
                        min_cycle=self.parameters.get('min_cycle', 13),
                        max_output=self.parameters.get('max_output', 124),
                        min_output=self.parameters.get('min_output', 13),
                        lp_period=self.parameters.get('lp_period', 13),
                        hp_period=self.parameters.get('hp_period', 124)
                    )
                else:
                    return ALMA(
                        length=self.parameters.get('length', 9),
                        offset=self.parameters.get('offset', 0.85),
                        sigma=self.parameters.get('sigma', 6.0),
                        src_type=self.src_type
                    )
            
            elif self.smoother_type == 'hma':
                # HMA - Hull Moving Average
                if self.period_mode == 'dynamic':
                    return HMA(
                        length=self.parameters.get('length', 14),
                        src_type=self.src_type,
                        period_mode='dynamic',
                        cycle_detector_type=self.parameters.get('cycle_detector_type', 'hody_e'),
                        cycle_part=self.parameters.get('cycle_part', 0.5),
                        max_cycle=self.parameters.get('max_cycle', 124),
                        min_cycle=self.parameters.get('min_cycle', 13),
                        max_output=self.parameters.get('max_output', 124),
                        min_output=self.parameters.get('min_output', 13),
                        lp_period=self.parameters.get('lp_period', 13),
                        hp_period=self.parameters.get('hp_period', 124)
                    )
                else:
                    return HMA(
                        length=self.parameters.get('length', 14),
                        src_type=self.src_type
                    )
            
            elif self.smoother_type == 'ultimate_ma':
                # Ultimate MA
                if UltimateMA is not None:
                    return UltimateMA(
                        ultimate_smoother_period=self.parameters.get('ultimate_smoother_period', 5.0),
                        zero_lag_period=self.parameters.get('zero_lag_period', 21),
                        realtime_window=self.parameters.get('realtime_window', 89),
                        src_type=self.src_type,
                        slope_index=self.parameters.get('slope_index', 1),
                        range_threshold=self.parameters.get('range_threshold', 0.005),
                        use_adaptive_kalman=self.parameters.get('use_adaptive_kalman', True),
                        zero_lag_period_mode=self.parameters.get('zero_lag_period_mode', 'fixed'),
                        realtime_window_mode=self.parameters.get('realtime_window_mode', 'fixed')
                    )
                else:
                    raise ImportError("Ultimate MA が利用できません。ライブラリがインポートされていない可能性があります。")
            
            else:
                # 汎用コンストラクタ
                return smoother_class(src_type=self.src_type, **self.parameters)
                
        except Exception as e:
            self.logger.error(f"スムーサーインスタンス作成エラー ({self.smoother_type}): {e}")
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
            
            params_sig = f"{self.smoother_type}_{self.src_type}_{hash(str(sorted(self.parameters.items())))}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.smoother_type}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UnifiedSmootherResult:
        """
        統合スムーサーを使用してスムージングを計算
        
        処理フロー: 価格データ → カルマンフィルター（オプション） → スムーサー
        
        Args:
            data: 価格データ
            
        Returns:
            UnifiedSmootherResult: 計算結果
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
                return self._copy_result(cached_result)
            
            # ステップ1: 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < 2:
                return self._create_empty_result(data_length, src_prices)
            
            # ステップ2: カルマンフィルターの適用（オプション）
            kalman_filtered_values = None
            processed_data = data
            
            if self.enable_kalman and self.kalman_filter is not None:
                # カルマンフィルターを適用
                kalman_result = self.kalman_filter.calculate(data)
                kalman_filtered_values = kalman_result.values
                
                # フィルター済み値で新しいデータフレーム/配列を構築
                processed_data = self._create_filtered_data(data, kalman_filtered_values)
            
            # ステップ3: スムーサーの計算実行
            smoother_result = self.smoother.calculate(processed_data)
            
            # 結果の標準化
            if hasattr(smoother_result, 'values'):
                # 構造化された結果の場合
                smoothed_values = smoother_result.values
                additional_data = self._extract_additional_data(smoother_result)
            elif hasattr(smoother_result, '_asdict'):  # NamedTuple（Ultimate MA）
                smoothed_values = smoother_result.values
                additional_data = self._extract_additional_data(smoother_result)
            else:
                # NumPy配列の場合
                smoothed_values = smoother_result
                additional_data = {}
            
            # NumPy配列への変換（必要に応じて）
            if not isinstance(smoothed_values, np.ndarray):
                smoothed_values = np.array(smoothed_values)
            
            # 結果の作成
            result = UnifiedSmootherResult(
                values=smoothed_values.copy(),
                raw_values=src_prices.copy(),
                kalman_filtered_values=kalman_filtered_values.copy() if kalman_filtered_values is not None else None,
                smoother_type=self.smoother_type,
                kalman_type=self.kalman_type,
                parameters=self.parameters.copy(),
                kalman_parameters=self.kalman_parameters.copy(),
                additional_data=additional_data
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = smoothed_values
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"統合スムーサー計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _extract_additional_data(self, smoother_result) -> Dict[str, np.ndarray]:
        """スムーサー固有の追加データを抽出"""
        additional_data = {}
        
        try:
            # FRAMA固有データ
            if hasattr(smoother_result, 'fractal_dimension'):
                additional_data['fractal_dimension'] = smoother_result.fractal_dimension.copy()
            if hasattr(smoother_result, 'alpha'):
                additional_data['alpha'] = smoother_result.alpha.copy()
            
            # ZLEMA固有データ
            if hasattr(smoother_result, 'ema_values'):
                additional_data['ema_values'] = smoother_result.ema_values.copy()
            if hasattr(smoother_result, 'lag_reduced_data'):
                additional_data['lag_reduced_data'] = smoother_result.lag_reduced_data.copy()
            
            # ALMA固有データ
            if hasattr(smoother_result, 'weights'):
                additional_data['alma_weights'] = smoother_result.weights.copy()
            
            # HMA固有データ
            if hasattr(smoother_result, 'wma1_values'):
                additional_data['hma_wma1'] = smoother_result.wma1_values.copy()
            if hasattr(smoother_result, 'wma2_values'):
                additional_data['hma_wma2'] = smoother_result.wma2_values.copy()
            if hasattr(smoother_result, 'diff_values'):
                additional_data['hma_diff'] = smoother_result.diff_values.copy()
            
            # Ultimate MA 固有データ
            if hasattr(smoother_result, 'raw_values'):
                additional_data['ultimate_ma_raw_values'] = smoother_result.raw_values.copy()
            if hasattr(smoother_result, 'ukf_values'):
                additional_data['ultimate_ma_ukf_values'] = smoother_result.ukf_values.copy()
            if hasattr(smoother_result, 'kalman_values'):
                additional_data['ultimate_ma_kalman_values'] = smoother_result.kalman_values.copy()
            if hasattr(smoother_result, 'kalman_gains'):
                additional_data['ultimate_ma_kalman_gains'] = smoother_result.kalman_gains.copy()
            if hasattr(smoother_result, 'kalman_innovations'):
                additional_data['ultimate_ma_kalman_innovations'] = smoother_result.kalman_innovations.copy()
            if hasattr(smoother_result, 'kalman_confidence'):
                additional_data['ultimate_ma_kalman_confidence'] = smoother_result.kalman_confidence.copy()
            if hasattr(smoother_result, 'ultimate_smooth_values'):
                additional_data['ultimate_ma_ultimate_smooth_values'] = smoother_result.ultimate_smooth_values.copy()
            if hasattr(smoother_result, 'zero_lag_values'):
                additional_data['ultimate_ma_zero_lag_values'] = smoother_result.zero_lag_values.copy()
            if hasattr(smoother_result, 'amplitude'):
                additional_data['ultimate_ma_amplitude'] = smoother_result.amplitude.copy()
            if hasattr(smoother_result, 'phase'):
                additional_data['ultimate_ma_phase'] = smoother_result.phase.copy()
            if hasattr(smoother_result, 'realtime_trends'):
                additional_data['ultimate_ma_realtime_trends'] = smoother_result.realtime_trends.copy()
            if hasattr(smoother_result, 'trend_signals'):
                additional_data['ultimate_ma_trend_signals'] = smoother_result.trend_signals.copy()
            if hasattr(smoother_result, 'current_trend'):
                additional_data['ultimate_ma_current_trend'] = smoother_result.current_trend
            if hasattr(smoother_result, 'current_trend_value'):
                additional_data['ultimate_ma_current_trend_value'] = smoother_result.current_trend_value
            
        except Exception as e:
            self.logger.warning(f"追加データ抽出中にエラー: {e}")
        
        return additional_data
    
    def _create_filtered_data(self, data: Union[pd.DataFrame, np.ndarray], kalman_filtered_values: np.ndarray) -> Union[pd.DataFrame, np.ndarray]:
        """カルマンフィルター適用後のデータを構築"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合: ソース価格をフィルター済み値で置換
            filtered_data = data.copy()
            filtered_data[self.src_type] = kalman_filtered_values
            
            # 論理的整合性を保持（他のOHLV価格も比例調整）
            if self.src_type == 'close':
                for i in range(len(filtered_data)):
                    if not np.isnan(kalman_filtered_values[i]) and not np.isnan(data.iloc[i]['close']) and data.iloc[i]['close'] != 0:
                        ratio = kalman_filtered_values[i] / data.iloc[i]['close']
                        for col in ['open', 'high', 'low']:
                            if col in filtered_data.columns:
                                filtered_data.iloc[i, filtered_data.columns.get_loc(col)] *= ratio
            
            return filtered_data
        else:
            # NumPy配列の場合
            filtered_data = data.copy()
            if filtered_data.ndim > 1 and filtered_data.shape[1] >= 4:
                # OHLCV形式の場合、適切な列にフィルター済み値を設定
                if self.src_type == 'close':
                    filtered_data[:, 3] = kalman_filtered_values  # close列
                elif self.src_type == 'open':
                    filtered_data[:, 0] = kalman_filtered_values  # open列
                elif self.src_type == 'high':
                    filtered_data[:, 1] = kalman_filtered_values  # high列
                elif self.src_type == 'low':
                    filtered_data[:, 2] = kalman_filtered_values  # low列
                else:
                    # hlc3, hl2, ohlc4などの複合価格の場合は全体を調整
                    filtered_data[:, 3] = kalman_filtered_values  # close列をベースとして使用
            else:
                # 1次元配列の場合はそのまま置換
                filtered_data = kalman_filtered_values
            
            return filtered_data
    
    def _copy_result(self, result: UnifiedSmootherResult) -> UnifiedSmootherResult:
        """結果をコピー"""
        return UnifiedSmootherResult(
            values=result.values.copy(),
            raw_values=result.raw_values.copy(),
            kalman_filtered_values=result.kalman_filtered_values.copy() if result.kalman_filtered_values is not None else None,
            smoother_type=result.smoother_type,
            kalman_type=result.kalman_type,
            parameters=result.parameters.copy(),
            kalman_parameters=result.kalman_parameters.copy(),
            additional_data={k: v.copy() if isinstance(v, np.ndarray) else v for k, v in result.additional_data.items()}
        )
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> UnifiedSmootherResult:
        """空の結果を作成"""
        return UnifiedSmootherResult(
            values=np.full(length, np.nan),
            raw_values=raw_prices,
            kalman_filtered_values=None,
            smoother_type=self.smoother_type,
            kalman_type=self.kalman_type,
            parameters=self.parameters.copy(),
            kalman_parameters=self.kalman_parameters.copy(),
            additional_data={}
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """スムースされた値を取得"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_raw_values(self) -> Optional[np.ndarray]:
        """元の価格データを取得"""
        result = self._get_latest_result()
        return result.raw_values.copy() if result else None
    
    def get_additional_data(self, key: str) -> Optional[np.ndarray]:
        """スムーサー固有の追加データを取得"""
        result = self._get_latest_result()
        if result and key in result.additional_data:
            return result.additional_data[key].copy()
        return None
    
    def get_smoother_info(self) -> Dict[str, Any]:
        """スムーサー情報を取得"""
        return {
            'type': self.smoother_type,
            'description': self._SMOOTHER_DESCRIPTIONS.get(self.smoother_type, 'Unknown'),
            'src_type': self.src_type,
            'parameters': self.parameters.copy()
        }
    
    def _get_latest_result(self) -> Optional[UnifiedSmootherResult]:
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
        if hasattr(self.smoother, 'reset'):
            self.smoother.reset()
        self._result_cache = {}
        self._cache_keys = []
    
    @classmethod
    def get_available_smoothers(cls) -> Dict[str, str]:
        """
        利用可能なスムーサーとその説明を返す
        
        Returns:
            Dict[str, str]: スムーサー名とその説明の辞書
        """
        return cls._SMOOTHER_DESCRIPTIONS.copy()
    
    @classmethod
    def get_default_parameters(cls, smoother_type: str) -> Dict[str, Any]:
        """
        指定されたスムーサーのデフォルトパラメータを取得
        
        Args:
            smoother_type: スムーサータイプ
            
        Returns:
            Dict[str, Any]: デフォルトパラメータ
        """
        smoother_type = smoother_type.lower()
        return cls._DEFAULT_PARAMS.get(smoother_type, {}).copy()


# 便利関数
def smooth(
    data: Union[pd.DataFrame, np.ndarray], 
    smoother_type: str = 'frama',
    src_type: str = 'close',
    **kwargs
) -> np.ndarray:
    """
    統合スムーサーの計算（便利関数）
    
    Args:
        data: 価格データ
        smoother_type: スムーサータイプ
        src_type: 価格ソース
        **kwargs: スムーサー固有のパラメータ
        
    Returns:
        スムースされた値
    """
    smoother = UnifiedSmoother(smoother_type=smoother_type, src_type=src_type, **kwargs)
    result = smoother.calculate(data)
    return result.values


def compare_smoothers(
    data: Union[pd.DataFrame, np.ndarray],
    smoother_types: list = ['frama', 'super_smoother', 'zero_lag_ema'],
    src_type: str = 'close',
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    複数のスムーサーを比較
    
    Args:
        data: 価格データ
        smoother_types: 比較するスムーサータイプのリスト
        src_type: 価格ソース
        **kwargs: スムーサー固有のパラメータ
        
    Returns:
        Dict[str, np.ndarray]: スムーサータイプ別の結果
    """
    results = {}
    
    for smoother_type in smoother_types:
        try:
            smoother = UnifiedSmoother(smoother_type=smoother_type, src_type=src_type, **kwargs)
            result = smoother.calculate(data)
            results[smoother_type] = result.values
        except Exception as e:
            print(f"Error with {smoother_type}: {e}")
            results[smoother_type] = np.full(len(data), np.nan)
    
    return results


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("=== 統合スムーサーのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 100
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
    
    # 利用可能なスムーサーを表示
    smoothers = UnifiedSmoother.get_available_smoothers()
    print(f"\n利用可能なスムーサー: {len(smoothers)}種類")
    for name, desc in list(smoothers.items())[:5]:  # 最初の5個のみ表示
        print(f"  {name}: {desc}")
    
    # 各スムーサーをテスト
    test_smoothers = ['frama', 'zero_lag_ema', 'super_smoother']
    print(f"\nテスト対象: {test_smoothers}")
    
    for smoother_type in test_smoothers:
        try:
            print(f"\n{smoother_type} をテスト中...")
            smoother = UnifiedSmoother(smoother_type=smoother_type, src_type='close')
            result = smoother.calculate(df)
            
            mean_smoothed = np.nanmean(result.values)
            mean_raw = np.nanmean(result.raw_values)
            valid_count = np.sum(~np.isnan(result.values))
            
            print(f"  平均値: {mean_smoothed:.4f} (元: {mean_raw:.4f})")
            print(f"  有効値数: {valid_count}/{len(df)}")
            print(f"  追加データ: {list(result.additional_data.keys()) if result.additional_data else 'なし'}")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n=== テスト完了 ===")