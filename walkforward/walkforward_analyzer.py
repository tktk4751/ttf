#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from analytics.analytics import Analytics
from .walkforward_optimizer import IWalkForwardResult

class WalkForwardAnalyzer:
    """ウォークフォワードテストの結果を分析するクラス"""
    
    def __init__(self, initial_balance: float):
        """
        Args:
            initial_balance: 初期資金
        """
        self.initial_balance = initial_balance
    
    def _analyze_parameter_stability(self, period_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """パラメータの安定性を分析
        
        Args:
            period_results: 期間ごとの結果リスト
        
        Returns:
            パラメータごとの統計情報
        """
        # 各期間のパラメータを収集
        param_values = {}
        for period in period_results:
            for param_name, param_value in period['parameters'].items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)
        
        # パラメータごとの統計を計算
        stability_metrics = {}
        for param_name, values in param_values.items():
            values = np.array(values)
            stability_metrics[param_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'cv': float(np.std(values) / np.mean(values))  # 変動係数
            }
        
        return stability_metrics

    def analyze(self, result: IWalkForwardResult) -> Dict[str, Any]:
        """結果を分析
        
        Args:
            result: ウォークフォワードテストの結果
        
        Returns:
            分析結果の辞書
        """
        # 全期間の分析
        all_trades_analytics = Analytics(result.trades, self.initial_balance)
        overall_metrics = all_trades_analytics.get_full_analysis()
        
        # 期間ごとの分析
        period_metrics = []
        for period in result.period_results:
            analytics = Analytics(period['trades'], self.initial_balance)
            metrics = analytics.get_full_analysis()
            metrics.update({
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'parameters': period['parameters'],
                'training_score': period['training_score']
            })
            period_metrics.append(metrics)
        
        # パラメータの安定性分析
        param_stability = self._analyze_parameter_stability(result.period_results)
        
        return {
            'overall_metrics': overall_metrics,
            'period_metrics': period_metrics,
            'parameter_stability': param_stability
        }
    
    def print_results(self, analysis_results: Dict[str, Any]):
        """分析結果を出力
        
        Args:
            analysis_results: analyze()メソッドの戻り値
        """
        print("\n=== ウォークフォワードテスト結果 ===")
        
        # 全体の結果
        print("\n全期間の結果:")
        metrics = analysis_results['overall_metrics']
        print(f"総トレード数: {metrics['total_trades']}")
        print(f"勝率: {metrics['win_rate']:.2f}%")
        print(f"プロフィットファクター: {metrics['profit_factor']:.2f}")
        print(f"最大ドローダウン: {metrics['max_drawdown']:.2f}%")
        print(f"シャープレシオ: {metrics['sharpe_ratio']:.2f}")
        
        # 期間ごとの結果
        print("\n期間ごとの結果:")
        for period in analysis_results['period_metrics']:
            print(f"\n期間: {period['start_date']} - {period['end_date']}")
            print(f"トレード数: {period['total_trades']}")
            print(f"勝率: {period['win_rate']:.2f}%")
            print(f"プロフィットファクター: {period['profit_factor']:.2f}")
            print(f"最大ドローダウン: {period['max_drawdown']:.2f}%")
        
        # パラメータの安定性
        print("\nパラメータの安定性:")
        for param_name, stats in analysis_results['parameter_stability'].items():
            print(f"\n{param_name}:")
            print(f"  平均: {stats['mean']:.2f}")
            print(f"  標準偏差: {stats['std']:.2f}")
            print(f"  変動係数: {stats['cv']:.2f}")
            print(f"  範囲: {stats['min']:.2f} - {stats['max']:.2f}") 