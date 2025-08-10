#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path

# TTFシステムのインポート
sys.path.append(str(Path(__file__).parent.parent.parent))


class TrendFollowEvaluation:
    """
    トレンドフォローモデル評価クラス
    
    仕様書通りの実装:
    - 多クラス分類性能評価
    - トレーディング特化指標
    - リスク評価指標
    - 詳細な分析とレポート生成
    """
    
    def __init__(self):
        """初期化"""
        self.class_names = ['失敗・中立', '買い成功', '売り成功']
        self.class_mapping = {0: '失敗・中立', 1: '買い成功', 2: '売り成功'}
    
    def evaluate_classification_performance(self, 
                                          y_true: np.ndarray, 
                                          y_pred: np.ndarray,
                                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        多クラス分類性能評価
        
        Args:
            y_true: 実際のクラスラベル
            y_pred: 予測クラスラベル
            y_pred_proba: 予測確率（オプション）
        
        Returns:
            分類性能の評価結果
        """
        print("=== 多クラス分類性能評価 ===")
        
        # 基本指標計算
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # マクロ・ウェイト平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        
        # クラス別詳細
        class_details = {}
        for i, class_name in enumerate(self.class_names):
            class_details[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            }
        
        # 多クラスAUC（マクロ平均）
        multiclass_auc = None
        if y_pred_proba is not None and y_pred_proba.shape[1] >= 3:
            try:
                multiclass_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"AUC計算エラー: {str(e)}")
        
        # ログ損失
        log_loss_score = None
        if y_pred_proba is not None:
            try:
                log_loss_score = log_loss(y_true, y_pred_proba)
            except Exception as e:
                print(f"Log Loss計算エラー: {str(e)}")
        
        result = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'class_details': class_details,
            'confusion_matrix': cm.tolist(),
            'multiclass_auc': multiclass_auc,
            'log_loss': log_loss_score,
            'classification_report': classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        }
        
        # 結果表示
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        if multiclass_auc:
            print(f"Multiclass AUC: {multiclass_auc:.4f}")
        
        return result
    
    def evaluate_trading_performance(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   original_targets: np.ndarray) -> Dict[str, Any]:
        """
        トレーディング特化指標評価
        
        Args:
            y_true: 実際のクラスラベル
            y_pred: 予測クラスラベル
            original_targets: 元の目的変数 (+1, -1, 0)
        
        Returns:
            トレーディング性能の評価結果
        """
        print("\n=== トレーディング特化指標評価 ===")
        
        # シグナル精度（成功シグナルの予測精度）
        success_mask = (y_true == 1) | (y_true == 2)  # 成功シグナル
        if success_mask.sum() > 0:
            signal_accuracy = accuracy_score(y_true[success_mask], y_pred[success_mask])
        else:
            signal_accuracy = 0.0
        
        # 方向性精度（買い・売りの方向が正しいか）
        buy_mask = y_true == 1
        sell_mask = y_true == 2
        
        buy_accuracy = accuracy_score(y_true[buy_mask], y_pred[buy_mask]) if buy_mask.sum() > 0 else 0.0
        sell_accuracy = accuracy_score(y_true[sell_mask], y_pred[sell_mask]) if sell_mask.sum() > 0 else 0.0
        
        # Hit Rate（利益目標達成率）
        # 予測で成功と判定したもののうち、実際に成功した割合
        pred_success_mask = (y_pred == 1) | (y_pred == 2)
        if pred_success_mask.sum() > 0:
            actual_success_in_pred = ((y_true == 1) | (y_true == 2))[pred_success_mask]
            hit_rate = actual_success_in_pred.mean()
        else:
            hit_rate = 0.0
        
        # False Positive Rate（失敗を成功と誤判定する率）
        failure_mask = y_true == 0
        if failure_mask.sum() > 0:
            false_positive_rate = (y_pred[failure_mask] != 0).mean()
        else:
            false_positive_rate = 0.0
        
        # 各クラスの精度
        buy_precision = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None)[0][0] if (y_pred == 1).sum() > 0 else 0.0
        sell_precision = precision_recall_fscore_support(y_true, y_pred, labels=[2], average=None)[0][0] if (y_pred == 2).sum() > 0 else 0.0
        
        # リスク調整後の成功率（ATRベース）
        # 元の目的変数での成功率
        atr_success_rate = (np.abs(original_targets) > 0).mean()
        
        result = {
            'signal_accuracy': signal_accuracy,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'hit_rate': hit_rate,
            'false_positive_rate': false_positive_rate,
            'buy_precision': buy_precision,
            'sell_precision': sell_precision,
            'atr_success_rate': atr_success_rate,
            'signal_counts': {
                'total_signals': len(y_true),
                'buy_signals': (y_true == 1).sum(),
                'sell_signals': (y_true == 2).sum(),
                'neutral_signals': (y_true == 0).sum()
            },
            'prediction_counts': {
                'predicted_buy': (y_pred == 1).sum(),
                'predicted_sell': (y_pred == 2).sum(),
                'predicted_neutral': (y_pred == 0).sum()
            }
        }
        
        # 結果表示
        print(f"Signal Accuracy: {signal_accuracy:.4f}")
        print(f"Hit Rate: {hit_rate:.4f}")
        print(f"Buy Precision: {buy_precision:.4f}")
        print(f"Sell Precision: {sell_precision:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        print(f"ATR Success Rate: {atr_success_rate:.4f}")
        
        return result
    
    def evaluate_risk_metrics(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            confidence_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        リスク評価指標
        
        Args:
            y_true: 実際のクラスラベル
            y_pred: 予測クラスラベル
            confidence_scores: 予測信頼度（オプション）
        
        Returns:
            リスク評価結果
        """
        print("\n=== リスク評価指標 ===")
        
        # 予測信頼度分析
        confidence_analysis = {}
        if confidence_scores is not None:
            # 信頼度別の精度
            high_conf_mask = confidence_scores > 0.7
            medium_conf_mask = (confidence_scores > 0.5) & (confidence_scores <= 0.7)
            low_conf_mask = confidence_scores <= 0.5
            
            high_conf_accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else 0.0
            medium_conf_accuracy = accuracy_score(y_true[medium_conf_mask], y_pred[medium_conf_mask]) if medium_conf_mask.sum() > 0 else 0.0
            low_conf_accuracy = accuracy_score(y_true[low_conf_mask], y_pred[low_conf_mask]) if low_conf_mask.sum() > 0 else 0.0
            
            confidence_analysis = {
                'high_confidence_accuracy': high_conf_accuracy,
                'medium_confidence_accuracy': medium_conf_accuracy,
                'low_confidence_accuracy': low_conf_accuracy,
                'high_confidence_count': high_conf_mask.sum(),
                'medium_confidence_count': medium_conf_mask.sum(),
                'low_confidence_count': low_conf_mask.sum()
            }
        
        # 連続失敗の分析
        consecutive_failures = self._analyze_consecutive_failures(y_true, y_pred)
        
        # 予測の安定性（連続する予測の変化率）
        prediction_stability = self._analyze_prediction_stability(y_pred)
        
        result = {
            'confidence_analysis': confidence_analysis,
            'consecutive_failures': consecutive_failures,
            'prediction_stability': prediction_stability
        }
        
        return result
    
    def _analyze_consecutive_failures(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """連続失敗の分析"""
        # 予測ミスの位置
        errors = (y_true != y_pred)
        
        # 連続失敗の長さを計算
        consecutive_lengths = []
        current_length = 0
        
        for error in errors:
            if error:
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            consecutive_lengths.append(current_length)
        
        if consecutive_lengths:
            return {
                'max_consecutive_failures': max(consecutive_lengths),
                'avg_consecutive_failures': np.mean(consecutive_lengths),
                'total_failure_streaks': len(consecutive_lengths),
                'failure_streak_distribution': np.bincount(consecutive_lengths).tolist()
            }
        else:
            return {
                'max_consecutive_failures': 0,
                'avg_consecutive_failures': 0,
                'total_failure_streaks': 0,
                'failure_streak_distribution': []
            }
    
    def _analyze_prediction_stability(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """予測の安定性分析"""
        # 予測変化の回数
        changes = np.sum(y_pred[1:] != y_pred[:-1])
        change_rate = changes / len(y_pred) if len(y_pred) > 1 else 0.0
        
        # 各クラスの連続性
        class_runs = {}
        for class_id in [0, 1, 2]:
            mask = (y_pred == class_id)
            runs = []
            current_run = 0
            
            for is_class in mask:
                if is_class:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            
            if current_run > 0:
                runs.append(current_run)
            
            class_runs[self.class_mapping[class_id]] = {
                'avg_run_length': np.mean(runs) if runs else 0,
                'max_run_length': max(runs) if runs else 0,
                'total_runs': len(runs)
            }
        
        return {
            'prediction_change_rate': change_rate,
            'total_changes': changes,
            'class_run_analysis': class_runs
        }
    
    def generate_comprehensive_report(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    original_targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        包括的な評価レポート生成
        
        Args:
            y_true: 実際のクラスラベル
            y_pred: 予測クラスラベル
            y_pred_proba: 予測確率（オプション）
            original_targets: 元の目的変数（オプション）
        
        Returns:
            包括的評価結果
        """
        print("=== 包括的評価レポート生成 ===")
        
        # 各種評価実行
        classification_result = self.evaluate_classification_performance(y_true, y_pred, y_pred_proba)
        
        trading_result = {}
        if original_targets is not None:
            trading_result = self.evaluate_trading_performance(y_true, y_pred, original_targets)
        
        confidence_scores = None
        if y_pred_proba is not None:
            confidence_scores = np.max(y_pred_proba, axis=1)
        risk_result = self.evaluate_risk_metrics(y_true, y_pred, confidence_scores)
        
        # 統合レポート
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(y_true),
                'class_distribution': np.bincount(y_true).tolist(),
                'prediction_distribution': np.bincount(y_pred).tolist()
            },
            'classification_performance': classification_result,
            'trading_performance': trading_result,
            'risk_metrics': risk_result
        }
        
        return report
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None) -> None:
        """
        混同行列の可視化
        
        Args:
            y_true: 実際のクラスラベル
            y_pred: 予測クラスラベル
            save_path: 保存パス（オプション）
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混同行列を保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_performance(self, classification_result: Dict[str, Any],
                             save_path: Optional[str] = None) -> None:
        """
        クラス別性能の可視化
        
        Args:
            classification_result: 分類性能結果
            save_path: 保存パス（オプション）
        """
        class_details = classification_result['class_details']
        
        metrics = ['precision', 'recall', 'f1_score']
        classes = list(class_details.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [class_details[cls][metric] for cls in classes]
            axes[i].bar(classes, values)
            axes[i].set_title(f'{metric.title()}')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"クラス別性能チャートを保存: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """メイン実行関数"""
    # 評価器初期化
    evaluator = TrendFollowEvaluation()
    
    print("=== トレンドフォローモデル評価器 ===")
    print("クラス定義:")
    for i, name in enumerate(evaluator.class_names):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()