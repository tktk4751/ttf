#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import warnings
import sys
from pathlib import Path

# TTFシステムのインポート
sys.path.append(str(Path(__file__).parent.parent.parent))
from indicators.hyper_mama import HyperMAMA
from indicators.hyper_efficiency_ratio import HyperEfficiencyRatio
from indicators.smoother.frama import FRAMA
from indicators.trend_filter.phasor_trend_filter import PhasorTrendFilter
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.volatility.x_atr import XATR


class TrendFollowFeatureEngineering:
    """
    トレンドフォローモデル用特徴量エンジニアリング
    
    仕様書通りの実装:
    - 6つのコアインジケーター
    - 正規化処理（終値で除算）
    - 8期間の差分特徴量
    - 合計54次元の特徴空間
    """
    
    def __init__(self):
        """初期化"""
        self.lag_periods = [1, 3, 5, 8, 34, 55, 89, 144]
        self.features_cache = {}
        
        # インジケーターの初期化
        self._init_indicators()
    
    def _init_indicators(self):
        """インジケーターを初期化"""
        try:
            # 1. ハイパーMAMA（実際のパラメータ）
            self.hyper_mama = HyperMAMA(
                trigger_type='hyper_er',
                hyper_er_period=14,
                hyper_er_midline_period=100,
                hyper_er_er_period=13,
                hyper_er_src_type='oc2',
                fast_max=0.5,
                fast_min=0.1,
                slow_max=0.05,
                slow_min=0.01,
                er_high_threshold=0.8,
                er_low_threshold=0.2,
                src_type='hlc3',
                use_kalman_filter=False
            )
            
            # 2. ハイパーER（実際のパラメータ）
            self.hyper_er = HyperEfficiencyRatio(
                window=14,
                src_type='hlc3',
                use_dynamic_period=False,
                slope_index=3,
                threshold=0.3
            )
            
            # 3. FRAMA（実際のパラメータ）
            self.frama = FRAMA(
                period=16,
                src_type='hl2',
                fc=1,
                sc=198,
                period_mode='fixed',
                cycle_detector_type='hody_e'
            )
            
            # 4. フェーザートレンドフィルター（実際のパラメータ）
            self.phasor_trend = PhasorTrendFilter(
                period=20,
                trend_threshold=6.0,
                src_type='close',
                use_kalman_filter=True,
                use_dynamic_period=True,
                detector_type='dft_dominant',
                cycle_part=0.5,
                max_cycle=89,
                min_cycle=12
            )
            
            # 5. ハイパートレンドインデックス（実際のパラメータ）
            self.hyper_trend_idx = HyperTrendIndex(
                period=14,
                midline_period=100,
                src_type='hlc3',
                use_kalman_filter=True,
                kalman_filter_type='adaptive',
                use_dynamic_period=True,
                detector_type='hody_e',
                use_roofing_filter=True,
                use_smoothing=True,
                smoother_type='frama',
                smoother_period=12
            )
            
            # 6. X_ATR（実際のパラメータ）
            self.x_atr = XATR(
                period=20.0,
                tr_method='atr',
                smoother_type='ultimate_smoother',
                src_type='close',
                enable_kalman=False,
                period_mode='dynamic',
                cycle_detector_type='absolute_ultimate',
                midline_period=100,
                enable_percentile_analysis=True
            )
            
            print("インジケーター初期化完了")
            
        except Exception as e:
            print(f"インジケーター初期化エラー: {str(e)}")
            raise
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        全ての特徴量を計算
        
        Args:
            data: OHLCVデータフレーム
        
        Returns:
            特徴量データフレーム（54次元）
        """
        print("特徴量計算開始...")
        
        # データの妥当性チェック
        if not self._validate_input_data(data):
            raise ValueError("入力データが無効です")
        
        # データの先頭部分をスキップ（インジケーター安定化のため）
        skip_rows = max(self.lag_periods) + 200  # ラグ期間 + 安定化期間
        if len(data) <= skip_rows:
            raise ValueError(f"データが不足しています。最低{skip_rows}行必要です")
        
        stable_data = data.iloc[skip_rows:].copy()
        print(f"  安定化のため先頭{skip_rows}行をスキップ、残り{len(stable_data)}行で処理")
        
        features = pd.DataFrame(index=stable_data.index)
        
        # 1. コアインジケーター計算（安定化されたデータで）
        core_features = self._calculate_core_indicators(stable_data)
        features = pd.concat([features, core_features], axis=1)
        
        # 2. 差分特徴量生成
        diff_features = self._calculate_diff_features(core_features)
        features = pd.concat([features, diff_features], axis=1)
        
        # 3. NaN値の処理
        features = self._handle_nan_values(features)
        
        print(f"特徴量計算完了: {features.shape[1]}次元")
        return features
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """入力データの妥当性をチェック"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            print("エラー: 必要なカラムが不足しています")
            return False
        
        if len(data) < max(self.lag_periods) + 200:
            print(f"警告: データが少なすぎます。最低{max(self.lag_periods) + 200}行必要です")
            return False
        
        return True
    
    def _calculate_core_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        コアインジケーターを計算
        
        Args:
            data: 価格データ
        
        Returns:
            コアインジケーター特徴量
        """
        core_features = pd.DataFrame(index=data.index)
        
        try:
            # 1. ハイパーMAMA計算（MAMA値のみ使用、仕様書通り6特徴量）
            print("  ハイパーMAMA計算中...")
            hyper_mama_result = self.hyper_mama.calculate(data)
            if hyper_mama_result is not None:
                # 正規化が必要（終値で除算）- MAMA値のみ使用
                core_features['mama_norm'] = hyper_mama_result.mama_values / data['close']
            else:
                print("  警告: ハイパーMAMA計算失敗")
                core_features['mama_norm'] = np.nan
            
            # 2. FRAMA計算
            print("  FRAMA計算中...")
            frama_result = self.frama.calculate(data)
            if frama_result is not None and hasattr(frama_result, 'values'):
                # 正規化が必要（終値で除算）
                core_features['frama_norm'] = frama_result.values / data['close']
            else:
                print("  警告: FRAMA計算失敗")
                core_features['frama_norm'] = np.nan
            
            # 3. X_ATR計算
            print("  X_ATR計算中...")
            x_atr_result = self.x_atr.calculate(data)
            if x_atr_result is not None and hasattr(x_atr_result, 'values'):
                # 正規化が必要（終値で除算）
                core_features['x_atr_norm'] = x_atr_result.values / data['close']
            else:
                print("  警告: X_ATR計算失敗")
                core_features['x_atr_norm'] = np.nan
            
            # 4. ハイパーER計算
            print("  ハイパーER計算中...")
            hyper_er_result = self.hyper_er.calculate(data)
            if hyper_er_result is not None and hasattr(hyper_er_result, 'values'):
                # 正規化不要（既に0-1範囲）
                core_features['hyper_er'] = hyper_er_result.values
            else:
                print("  警告: ハイパーER計算失敗")
                core_features['hyper_er'] = np.nan
            
            # 5. フェーザートレンド計算
            print("  フェーザートレンド計算中...")
            phasor_result = self.phasor_trend.calculate(data)
            if phasor_result is not None and hasattr(phasor_result, 'values'):
                # 正規化不要（0-1範囲のトレンド強度値）
                core_features['phasor_values'] = phasor_result.values
            else:
                print("  警告: フェーザートレンド計算失敗")
                core_features['phasor_values'] = np.nan
            
            # 6. ハイパートレンドインデックス計算
            print("  ハイパートレンドインデックス計算中...")
            hyper_trend_result = self.hyper_trend_idx.calculate(data)
            if hyper_trend_result is not None and hasattr(hyper_trend_result, 'values'):
                # 正規化不要（正規化済み）
                core_features['hyper_trend_idx'] = hyper_trend_result.values
            else:
                print("  警告: ハイパートレンドインデックス計算失敗")
                core_features['hyper_trend_idx'] = np.nan
            
        except Exception as e:
            print(f"インジケーター計算エラー: {str(e)}")
            # エラーの場合はNaN値で埋める（6つのコア特徴量）
            for col in ['mama_norm', 'frama_norm', 'x_atr_norm', 'hyper_er', 'phasor_values', 'hyper_trend_idx']:
                if col not in core_features.columns:
                    core_features[col] = np.nan
        
        print(f"  コアインジケーター計算完了: {len(core_features.columns)}個")
        return core_features
    
    def _calculate_diff_features(self, core_features: pd.DataFrame) -> pd.DataFrame:
        """
        差分特徴量を計算
        
        Args:
            core_features: コアインジケーター特徴量
        
        Returns:
            差分特徴量
        """
        print("  差分特徴量計算中...")
        diff_features = pd.DataFrame(index=core_features.index)
        
        # 各コアインジケーターについて差分を計算
        for col in core_features.columns:
            for lag in self.lag_periods:
                diff_col_name = f'{col}_diff_{lag}'
                diff_features[diff_col_name] = core_features[col] - core_features[col].shift(lag)
        
        print(f"  差分特徴量計算完了: {len(diff_features.columns)}個")
        return diff_features
    
    def _handle_nan_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        NaN値を処理
        
        Args:
            features: 特徴量データフレーム
        
        Returns:
            NaN値処理済み特徴量データフレーム
        """
        print("  NaN値処理中...")
        
        # NaN値の統計
        nan_counts = features.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            print(f"    NaN値検出: {total_nans}個")
            
            # 前方埋め + 後方埋め
            features_filled = features.fillna(method='ffill').fillna(method='bfill')
            
            # まだNaN値がある場合は0で埋める
            remaining_nans = features_filled.isnull().sum().sum()
            if remaining_nans > 0:
                print(f"    残存NaN値を0で埋めます: {remaining_nans}個")
                features_filled = features_filled.fillna(0)
        else:
            features_filled = features
        
        print("  NaN値処理完了")
        return features_filled
    
    def get_feature_names(self) -> List[str]:
        """特徴量名のリストを取得"""
        core_names = ['mama_norm', 'fama_norm', 'frama_norm', 'x_atr_norm', 'hyper_er', 'phasor_values', 'hyper_trend_idx']
        
        feature_names = core_names.copy()
        
        # 差分特徴量名を追加
        for core_name in core_names:
            for lag in self.lag_periods:
                feature_names.append(f'{core_name}_diff_{lag}')
        
        return feature_names
    
    def get_feature_info(self) -> Dict[str, Any]:
        """特徴量の情報を取得"""
        feature_names = self.get_feature_names()
        
        return {
            'total_features': len(feature_names),
            'core_features': 6,
            'diff_features': len(feature_names) - 6,
            'lag_periods': self.lag_periods,
            'feature_names': feature_names,
            'normalized_features': ['mama_norm', 'fama_norm', 'frama_norm', 'x_atr_norm'],
            'raw_features': ['hyper_er', 'phasor_values', 'hyper_trend_idx']
        }
    
    def analyze_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        特徴量の統計分析
        
        Args:
            features: 特徴量データフレーム
        
        Returns:
            統計情報
        """
        return {
            'shape': features.shape,
            'null_counts': features.isnull().sum().to_dict(),
            'statistics': features.describe().to_dict(),
            'correlations': features.corr().abs().max().to_dict()
        }


def main():
    """メイン実行関数"""
    # 特徴量エンジニアリングの初期化
    feature_eng = TrendFollowFeatureEngineering()
    
    # 特徴量情報の表示
    info = feature_eng.get_feature_info()
    print("=== 特徴量エンジニアリング情報 ===")
    print(f"総特徴量数: {info['total_features']}")
    print(f"コア特徴量: {info['core_features']}")
    print(f"差分特徴量: {info['diff_features']}")
    print(f"差分期間: {info['lag_periods']}")
    print(f"正規化特徴量: {info['normalized_features']}")
    print(f"生特徴量: {info['raw_features']}")


if __name__ == "__main__":
    main()