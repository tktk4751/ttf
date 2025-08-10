#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightGBM ATRリスク調整トレンドフォローモデル V1

仕様書通りの実装:
- 6つのコアインジケーター（54次元特徴空間）
- ATRベースリスク調整リターン目的変数
- 3ATR損切り・7ATR利益確定・300ローソク足制限
- 多クラス分類（買い成功・売り成功・失敗中立）
- 時系列交差検証・ウォークフォワード分析
- 包括的評価・バックテスト
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# パス設定
sys.path.append(str(Path(__file__).parent.parent.parent))

# 自作モジュールのインポート
from data_pipeline import TrendFollowDataPipeline
from feature_engineering import TrendFollowFeatureEngineering
from target_calculation import TrendFollowTargetCalculation
from model_training import TrendFollowModelTraining
from evaluation import TrendFollowEvaluation
from backtesting import TrendFollowBacktesting


class TrendFollowV1Pipeline:
    """
    トレンドフォローV1 統合パイプライン
    
    全コンポーネントを統合した完全な機械学習パイプライン
    """
    
    def __init__(self, 
                 config_path: str = "ml/trend_follow_v1/config/model_config.yaml",
                 main_config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: モデル設定ファイルのパス
            main_config_path: メインTTF設定ファイルのパス
        """
        self.config_path = config_path
        self.main_config_path = main_config_path
        self.config = None
        
        # 結果保存用
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # コンポーネント初期化
        self._init_components()
    
    def _init_components(self):
        """コンポーネントの初期化"""
        print("=== トレンドフォローV1パイプライン初期化 ===")
        
        # 設定読み込み
        self._load_config()
        
        # 各コンポーネント初期化
        self.data_pipeline = TrendFollowDataPipeline(self.main_config_path)
        self.feature_engineering = TrendFollowFeatureEngineering()
        self.target_calculation = TrendFollowTargetCalculation(
            atr_period=self.config['target']['atr_period'],
            stop_atr_mult=self.config['target']['stop_atr_multiplier'],
            target_atr_mult=self.config['target']['target_atr_multiplier'],
            max_bars=self.config['target']['max_holding_bars']
        )
        self.model_training = TrendFollowModelTraining(self.main_config_path)
        self.evaluation = TrendFollowEvaluation()
        self.backtesting = TrendFollowBacktesting(
            initial_capital=self.config['backtest']['initial_capital'],
            commission_rate=self.config['backtest']['commission_rate'],
            confidence_threshold=self.config['backtest']['confidence_threshold']
        )
        
        print("コンポーネント初期化完了")
    
    def _load_config(self):
        """設定ファイルの読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print(f"設定ファイル読み込み完了: {self.config_path}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        完全なパイプラインの実行
        
        Returns:
            実行結果
        """
        print("=== 完全パイプライン実行開始 ===")
        
        try:
            # 1. データ準備
            print("\n--- Phase 1: データ準備 ---")
            features, targets, labels = self._prepare_data()
            
            # 2. モデル訓練・評価
            print("\n--- Phase 2: モデル訓練・評価 ---")
            model_results = self._train_and_evaluate_model(features, labels)
            
            # 3. バックテスト
            print("\n--- Phase 3: バックテスト ---")
            if self.config['backtest']['enabled']:
                backtest_results = self._run_backtest(features, targets, labels)
            else:
                backtest_results = {}
            
            # 4. 結果保存
            print("\n--- Phase 4: 結果保存 ---")
            self._save_results(model_results, backtest_results)
            
            # 5. レポート生成
            print("\n--- Phase 5: レポート生成 ---")
            final_report = self._generate_final_report(model_results, backtest_results)
            
            print("\n=== パイプライン実行完了 ===")
            return final_report
            
        except Exception as e:
            print(f"パイプライン実行エラー: {str(e)}")
            raise
    
    def _prepare_data(self) -> tuple:
        """データ準備"""
        # 相場データ読み込み
        data = self.data_pipeline.load_market_data()
        
        # 特徴量計算
        print("特徴量計算中...")
        features = self.feature_engineering.calculate_all_features(data)
        
        # 目的変数計算（特徴量と同じインデックスで）
        print("目的変数計算中...")
        targets = self.target_calculation.calculate_atr_normalized_target(data)
        labels = self.target_calculation.create_target_labels(targets)
        
        # 特徴量と同じインデックスに合わせる
        if len(features) != len(labels):
            print(f"  インデックス調整: 特徴量{len(features)}行, ラベル{len(labels)}行")
            # 特徴量のインデックスに合わせてラベルを切り取り
            # pandas Seriesの場合は.loc、numpy arrayの場合は位置インデックス
            if hasattr(labels, 'loc'):
                labels = labels.loc[features.index]
                targets = targets.loc[features.index]
            else:
                # numpy arrayの場合、特徴量の開始・終了位置をデータ全体から特定
                # features.indexから元データでの位置を特定
                data_index = data.index
                start_pos = data_index.get_loc(features.index[0])
                end_pos = start_pos + len(features)
                labels = labels[start_pos:end_pos]
                targets = targets[start_pos:end_pos]
        
        # 結果保存
        self.results['data_info'] = {
            'data_shape': data.shape,
            'features_shape': features.shape,
            'target_distribution': np.bincount(labels).tolist(),
            'feature_names': features.columns.tolist()
        }
        
        print(f"データ準備完了: 特徴量{features.shape}, 目的変数{len(targets)}")
        return features, targets, labels
    
    def _train_and_evaluate_model(self, features: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """モデル訓練・評価"""
        results = {}
        
        # 時系列交差検証
        if self.config['cross_validation']['method'] == 'time_series_split':
            print("時系列交差検証実行中...")
            cv_results = self.model_training.time_series_split_validation(
                features, labels, self.config['cross_validation']['n_splits']
            )
            results['cross_validation'] = cv_results
        
        # 最終モデル訓練
        print("最終モデル訓練中...")
        model = self.model_training.train_final_model(features, labels)
        
        # 特徴量重要度分析
        print("特徴量重要度分析中...")
        importance_results = self.model_training.analyze_feature_importance()
        results['feature_importance'] = importance_results
        
        # ウォークフォワード分析（シンプル50-50分割）
        if self.config['walk_forward']['enabled']:
            print("ウォークフォワード分析実行中...")
            wf_result = self.model_training.walk_forward_analysis(features, labels)
            results['walk_forward'] = wf_result
        
        # モデル評価
        print("モデル評価中...")
        train_split = int(len(features) * 0.85)
        X_test = features.iloc[train_split:]
        y_test = labels[train_split:]
        
        test_pred_proba = model.predict_proba(X_test)
        test_pred = np.argmax(test_pred_proba, axis=1)
        
        evaluation_results = self.evaluation.generate_comprehensive_report(
            y_test, test_pred, test_pred_proba
        )
        results['evaluation'] = evaluation_results
        
        # 可視化
        if self.config['evaluation']['plot_confusion_matrix']:
            self._plot_confusion_matrix(y_test, test_pred)
        
        if self.config['evaluation']['plot_class_performance']:
            self._plot_class_performance(evaluation_results['classification_performance'])
        
        # モデル保存
        if self.config['model_save']['save_model']:
            model_path = self._save_model(model)
            results['model_path'] = model_path
        
        return results
    
    def _run_backtest(self, features: pd.DataFrame, targets: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """バックテスト実行"""
        # テストデータでバックテスト
        train_split = int(len(features) * 0.85)
        
        # 相場データ再取得（価格情報が必要）
        market_data = self.data_pipeline.load_market_data()
        test_data = market_data.iloc[train_split:]
        
        # モデル予測
        model = self.model_training.model
        X_test = features.iloc[train_split:]
        test_predictions = model.predict_proba(X_test)
        
        # バックテスト実行
        print("バックテスト実行中...")
        backtest_results = self.backtesting.run_backtest(test_data, test_predictions)
        
        # エクイティカーブ描画
        if self.config['backtest']['plot_equity_curve']:
            equity_path = self._get_results_path('equity_curve.png')
            self.backtesting.plot_equity_curve(equity_path)
        
        return backtest_results
    
    def _save_model(self, model) -> str:
        """モデル保存"""
        model_dir = Path(self.config['model_save']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"lightgbm_trend_follow_v1_{self.timestamp}.pkl"
        self.model_training.save_model(str(model_path))
        
        return str(model_path)
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """混同行列描画"""
        save_path = self._get_results_path('confusion_matrix.png')
        self.evaluation.plot_confusion_matrix(y_true, y_pred, save_path)
    
    def _plot_class_performance(self, classification_result: Dict[str, Any]):
        """クラス別性能描画"""
        save_path = self._get_results_path('class_performance.png')
        self.evaluation.plot_class_performance(classification_result, save_path)
    
    def _get_results_path(self, filename: str) -> str:
        """結果ファイルのパス生成"""
        results_dir = Path(self.config['evaluation']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        return str(results_dir / f"{self.timestamp}_{filename}")
    
    def _save_results(self, model_results: Dict[str, Any], backtest_results: Dict[str, Any]):
        """結果保存"""
        all_results = {
            'pipeline_info': {
                'timestamp': self.timestamp,
                'config': self.config,
                'data_info': self.results.get('data_info', {})
            },
            'model_results': model_results,
            'backtest_results': backtest_results
        }
        
        # JSON形式で保存
        results_path = self._get_results_path('complete_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"結果保存完了: {results_path}")
        self.results.update(all_results)
    
    def _generate_final_report(self, model_results: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """最終レポート生成"""
        print("最終レポート生成中...")
        
        # サマリー作成
        summary = {
            'execution_timestamp': self.timestamp,
            'pipeline_status': 'completed',
            'model_performance': {},
            'backtest_performance': {},
            'key_metrics': {}
        }
        
        # モデル性能サマリー
        if 'evaluation' in model_results:
            eval_result = model_results['evaluation']
            if 'classification_performance' in eval_result:
                cp = eval_result['classification_performance']
                summary['model_performance'] = {
                    'accuracy': cp.get('accuracy', 0),
                    'macro_f1': cp.get('macro_f1', 0),
                    'weighted_f1': cp.get('weighted_f1', 0)
                }
        
        # バックテスト性能サマリー
        if backtest_results and 'total_return' in backtest_results:
            summary['backtest_performance'] = {
                'total_return': backtest_results.get('total_return', 0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'total_trades': backtest_results.get('total_trades', 0)
            }
        
        # 重要指標
        summary['key_metrics'] = {
            'features_count': self.results.get('data_info', {}).get('features_shape', [0, 0])[1],
            'data_points': self.results.get('data_info', {}).get('features_shape', [0, 0])[0],
            'target_classes': len(np.unique(self.results.get('data_info', {}).get('target_distribution', []))),
        }
        
        # レポートファイル保存
        report_path = self._get_results_path('final_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"最終レポート保存: {report_path}")
        return summary


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("LightGBM ATRリスク調整トレンドフォローモデル V1")
    print("=" * 60)
    
    try:
        # パイプライン初期化・実行
        pipeline = TrendFollowV1Pipeline()
        final_report = pipeline.run_full_pipeline()
        
        # 結果サマリー表示
        print("\n" + "=" * 60)
        print("実行結果サマリー")
        print("=" * 60)
        
        # モデル性能
        model_perf = final_report.get('model_performance', {})
        print(f"モデル精度: {model_perf.get('accuracy', 0):.4f}")
        print(f"マクロF1スコア: {model_perf.get('macro_f1', 0):.4f}")
        print(f"重み付きF1スコア: {model_perf.get('weighted_f1', 0):.4f}")
        
        # バックテスト性能
        backtest_perf = final_report.get('backtest_performance', {})
        if backtest_perf:
            print(f"\nバックテスト結果:")
            print(f"総リターン: {backtest_perf.get('total_return', 0):.2%}")
            print(f"シャープレシオ: {backtest_perf.get('sharpe_ratio', 0):.4f}")
            print(f"最大ドローダウン: {backtest_perf.get('max_drawdown', 0):.2%}")
            print(f"勝率: {backtest_perf.get('win_rate', 0):.2%}")
            print(f"総トレード数: {backtest_perf.get('total_trades', 0)}")
        
        # キーメトリクス
        key_metrics = final_report.get('key_metrics', {})
        print(f"\nキーメトリクス:")
        print(f"特徴量数: {key_metrics.get('features_count', 0)}")
        print(f"データポイント数: {key_metrics.get('data_points', 0)}")
        print(f"目的変数クラス数: {key_metrics.get('target_classes', 0)}")
        
        print("\n" + "=" * 60)
        print("実行完了!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n実行エラー: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()