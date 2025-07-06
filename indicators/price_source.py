#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
価格ソース計算ユーティリティ（UKF統合版）
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple, TYPE_CHECKING
import warnings

# UKFモジュールのインポート
UnscentedKalmanFilter = None
UKFResult = None

try:
    from .unscented_kalman_filter import UnscentedKalmanFilter, UKFResult
except ImportError:
    try:
        from unscented_kalman_filter import UnscentedKalmanFilter, UKFResult
    except ImportError:
        pass

# インポートが実際に失敗した場合のみ警告（UKFが実際に動作しているため警告を無効化）
# if UnscentedKalmanFilter is None:
#     warnings.warn("無香料カルマンフィルターが利用できません")

# 型チェック時のみインポート
if TYPE_CHECKING:
    from .unscented_kalman_filter import UnscentedKalmanFilter, UKFResult


class PriceSource:
    """価格ソースの計算ユーティリティクラス（UKF統合版）"""
    
    # UKFフィルターのキャッシュ
    _ukf_cache = {}
    
    @staticmethod
    def calculate_source(
        data: Union[pd.DataFrame, np.ndarray], 
        src_type: str = 'close',
        ukf_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        指定されたソースタイプの価格データを計算
        
        Args:
            data: 価格データ
            src_type: ソースタイプ
                基本: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKF: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
            ukf_params: UKFパラメータ（オプション）
        
        Returns:
            計算された価格配列（必ずnp.ndarray）
        """
        src_type = src_type.lower()
        
        result = None
        
        # UKFソースタイプの処理
        if src_type.startswith('ukf'):
            result = PriceSource._calculate_ukf_source(data, src_type, ukf_params)
        # 従来のソースタイプ処理
        elif isinstance(data, pd.DataFrame):
            result = PriceSource._calculate_from_dataframe(data, src_type)
        elif isinstance(data, np.ndarray):
            result = PriceSource._calculate_from_array(data, src_type)
        else:
            raise ValueError("サポートされていないデータ型です")
        
        # 結果を確実にnp.ndarrayに変換
        if result is not None:
            if isinstance(result, pd.Series):
                result = result.values
            elif not isinstance(result, np.ndarray):
                result = np.asarray(result)
            
            # 型とサイズの確認
            if result.dtype == np.object_:
                result = result.astype(np.float64)
            elif not np.issubdtype(result.dtype, np.number):
                result = result.astype(np.float64)
            
            return result
        else:
            raise ValueError("価格データの計算に失敗しました")
    
    @staticmethod
    def _calculate_ukf_source(
        data: Union[pd.DataFrame, np.ndarray], 
        src_type: str,
        ukf_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        UKFベースの価格ソースを計算
        
        Args:
            data: 価格データ
            src_type: UKFソースタイプ
            ukf_params: UKFパラメータ
        
        Returns:
            UKFフィルター済み価格配列
        """
        if UnscentedKalmanFilter is None:
            raise ImportError("UnscentedKalmanFilterが利用できません")
        
        # UKFのベースソースを決定
        if src_type == 'ukf':
            base_source = 'close'
        elif src_type.startswith('ukf_'):
            base_source = src_type[4:]  # 'ukf_'を除去
        else:
            base_source = 'close'
        
        # デフォルトUKFパラメータ
        default_params = {
            'alpha': 0.001,
            'beta': 2.0,
            'kappa': 0.0,
            'process_noise_scale': 0.001,
            'volatility_window': 10,
            'adaptive_noise': True
        }
        
        # パラメータをマージ
        if ukf_params:
            default_params.update(ukf_params)
        
        # UKFフィルターのキャッシュキーを作成
        cache_key = f"{base_source}_{hash(frozenset(default_params.items()))}"
        
        # キャッシュからUKFフィルターを取得または作成
        if cache_key not in PriceSource._ukf_cache:
            PriceSource._ukf_cache[cache_key] = UnscentedKalmanFilter(
                src_type=base_source,
                alpha=default_params['alpha'],
                beta=default_params['beta'],
                kappa=default_params['kappa'],
                process_noise_scale=default_params['process_noise_scale'],
                volatility_window=default_params['volatility_window'],
                adaptive_noise=default_params['adaptive_noise']
            )
        
        ukf_filter = PriceSource._ukf_cache[cache_key]
        
        try:
            # UKFを計算
            ukf_result = ukf_filter.calculate(data)
            result = ukf_result.filtered_values
            
            # 確実にnp.ndarrayに変換
            if not isinstance(result, np.ndarray):
                result = np.asarray(result)
            
            return result
            
        except Exception as e:
            warnings.warn(f"UKF計算に失敗しました: {str(e)}。ベースソースを返します。")
            # フォールバック: ベースソースを直接計算（無限再帰を防ぐ）
            if isinstance(data, pd.DataFrame):
                return PriceSource._calculate_from_dataframe(data, base_source)
            elif isinstance(data, np.ndarray):
                return PriceSource._calculate_from_array(data, base_source)
            else:
                raise ValueError("フォールバック処理中にデータ型エラーが発生しました")
    
    @staticmethod
    def _calculate_from_dataframe(data: pd.DataFrame, src_type: str) -> np.ndarray:
        """DataFrameから価格を計算"""
        # カラム名のマッピング
        column_mapping = {
            'open': ['open', 'Open'],
            'high': ['high', 'High'], 
            'low': ['low', 'Low'],
            'close': ['close', 'Close', 'adj close', 'Adj Close']
        }
        
        # OHLCカラムを見つける
        ohlc = {}
        for key, possible_names in column_mapping.items():
            found = False
            for name in possible_names:
                if name in data.columns:
                    ohlc[key] = data[name].values
                    found = True
                    break
            
            # closeは常に必須（他のソースタイプでも必要）
            if not found and key == 'close':
                raise ValueError(f"'{key}' カラムが見つかりません")
        
        # ソースタイプに基づいて計算
        if src_type == 'close':
            return ohlc['close']
        elif src_type == 'high':
            if 'high' not in ohlc:
                raise ValueError("high カラムが見つかりません")
            return ohlc['high']
        elif src_type == 'low':
            if 'low' not in ohlc:
                raise ValueError("low カラムが見つかりません")
            return ohlc['low']
        elif src_type == 'open':
            if 'open' not in ohlc:
                raise ValueError("open カラムが見つかりません")
            return ohlc['open']
        elif src_type == 'hlc3':
            if 'high' not in ohlc or 'low' not in ohlc:
                raise ValueError("hlc3には high, low, close が必要です")
            return (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3.0
        elif src_type == 'hl2':
            if 'high' not in ohlc or 'low' not in ohlc:
                raise ValueError("hl2には high, low が必要です")
            return (ohlc['high'] + ohlc['low']) / 2.0
        elif src_type == 'ohlc4':
            if any(k not in ohlc for k in ['open', 'high', 'low']):
                raise ValueError("ohlc4には open, high, low, close が必要です")
            return (ohlc['open'] + ohlc['high'] + ohlc['low'] + ohlc['close']) / 4.0
        else:
            raise ValueError(f"サポートされていないソースタイプ: {src_type}")
    
    @staticmethod
    def _calculate_from_array(data: np.ndarray, src_type: str) -> np.ndarray:
        """NumPy配列から価格を計算"""
        if data.ndim == 1:
            if src_type not in ['close']:
                raise ValueError("1次元配列では'close'のみサポートされています")
            return data
        elif data.ndim == 2 and data.shape[1] >= 4:
            # OHLC形式を想定 [open, high, low, close]
            if src_type == 'close':
                return data[:, 3]
            elif src_type == 'open':
                return data[:, 0]
            elif src_type == 'high':
                return data[:, 1]
            elif src_type == 'low':
                return data[:, 2]
            elif src_type == 'hlc3':
                return (data[:, 1] + data[:, 2] + data[:, 3]) / 3.0
            elif src_type == 'hl2':
                return (data[:, 1] + data[:, 2]) / 2.0
            elif src_type == 'ohlc4':
                return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4.0
            else:
                raise ValueError(f"サポートされていないソースタイプ: {src_type}")
        else:
            raise ValueError("配列は1次元またはOHLC形式の2次元である必要があります")
    
    @staticmethod
    def get_available_sources() -> Dict[str, str]:
        """
        利用可能なソースタイプの一覧を取得
        
        Returns:
            ソースタイプ辞書 {タイプ: 説明}
        """
        sources = {
            'close': '終値',
            'open': '始値',
            'high': '高値',
            'low': '安値',
            'hlc3': '(高値 + 安値 + 終値) / 3',
            'hl2': '(高値 + 安値) / 2',
            'ohlc4': '(始値 + 高値 + 安値 + 終値) / 4'
        }
        
        # UKFが利用可能な場合は追加
        if UnscentedKalmanFilter is not None:
            ukf_sources = {
                'ukf': 'UKFフィルター済み終値',
                'ukf_close': 'UKFフィルター済み終値',
                'ukf_hlc3': 'UKFフィルター済みHLC3',
                'ukf_hl2': 'UKFフィルター済みHL2',
                'ukf_ohlc4': 'UKFフィルター済みOHLC4'
            }
            sources.update(ukf_sources)
        
        return sources
    
    @staticmethod
    def is_ukf_source(src_type: str) -> bool:
        """
        指定されたソースタイプがUKFベースかどうかを判定
        
        Args:
            src_type: ソースタイプ
        
        Returns:
            UKFベースの場合True
        """
        return src_type.lower().startswith('ukf')
    
    @staticmethod
    def get_ukf_result(
        data: Union[pd.DataFrame, np.ndarray], 
        src_type: str = 'close',
        ukf_params: Optional[Dict] = None
    ):
        """
        UKFの完全な結果を取得（フィルター値以外の情報も含む）
        
        Args:
            data: 価格データ
            src_type: ベースソースタイプ
            ukf_params: UKFパラメータ
        
        Returns:
            UKFResult: 完全なUKF結果（利用不可の場合はNone）
        """
        if UnscentedKalmanFilter is None:
            return None
        
        # デフォルトUKFパラメータ
        default_params = {
            'alpha': 0.001,
            'beta': 2.0,
            'kappa': 0.0,
            'process_noise_scale': 0.001,
            'volatility_window': 10,
            'adaptive_noise': True
        }
        
        if ukf_params:
            default_params.update(ukf_params)
        
        # UKFフィルターを作成
        ukf_filter = UnscentedKalmanFilter(
            src_type=src_type.lower(),
            alpha=default_params['alpha'],
            beta=default_params['beta'],
            kappa=default_params['kappa'],
            process_noise_scale=default_params['process_noise_scale'],
            volatility_window=default_params['volatility_window'],
            adaptive_noise=default_params['adaptive_noise']
        )
        
        try:
            return ukf_filter.calculate(data)
        except Exception:
            return None
    
    @staticmethod
    def clear_ukf_cache() -> None:
        """UKFキャッシュをクリア"""
        PriceSource._ukf_cache.clear() 