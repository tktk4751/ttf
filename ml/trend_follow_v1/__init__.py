#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightGBM ATRリスク調整トレンドフォローモデル V1

このパッケージは、TTF（Trading Technical Framework）システム内で
動作する高度な機械学習ベースのトレンドフォローモデルを提供します。

主な特徴:
- 6つの量子アルゴリズム系インジケーター（54次元特徴空間）
- ATRベースのリスク調整リターン目的変数
- 3ATR損切り・7ATR利益確定・300ローソク足制限
- LightGBM多クラス分類（買い成功・売り成功・失敗中立）
- 時系列交差検証・ウォークフォワード分析
- 包括的評価・リアルタイムバックテスト

使用方法:
    from ml.trend_follow_v1.main import TrendFollowV1Pipeline
    
    pipeline = TrendFollowV1Pipeline()
    results = pipeline.run_full_pipeline()
"""

from .main import TrendFollowV1Pipeline
from .data_pipeline import TrendFollowDataPipeline
from .feature_engineering import TrendFollowFeatureEngineering
from .target_calculation import TrendFollowTargetCalculation
from .model_training import TrendFollowModelTraining
from .evaluation import TrendFollowEvaluation
from .backtesting import TrendFollowBacktesting

__version__ = "1.0.0"
__author__ = "TTF Development Team"
__description__ = "LightGBM ATR Risk-Adjusted Trend Following Model"

__all__ = [
    "TrendFollowV1Pipeline",
    "TrendFollowDataPipeline", 
    "TrendFollowFeatureEngineering",
    "TrendFollowTargetCalculation",
    "TrendFollowModelTraining",
    "TrendFollowEvaluation",
    "TrendFollowBacktesting"
]