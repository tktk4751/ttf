#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import sys
from pathlib import Path

# TTFシステムのインポート
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_pipeline import TrendFollowDataPipeline
from feature_engineering import TrendFollowFeatureEngineering
from target_calculation import TrendFollowTargetCalculation


class TrendFollowModelTraining:
    """
    トレンドフォローモデル訓練クラス
    
    仕様書通りの実装:
    - LightGBM多クラス分類
    - 時系列交差検証
    - %ベースウォークフォワード分析
    - SHAP特徴量重要度分析
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        
        # コンポーネント初期化
        self.data_pipeline = TrendFollowDataPipeline(config_path)
        self.feature_engineering = TrendFollowFeatureEngineering()
        self.target_calculation = TrendFollowTargetCalculation()
        
        # モデル関連
        self.model = None
        self.feature_names = None
        self.training_history = []
        
        # LightGBMパラメータ（仕様書通り）
        self.lgb_params = {
            'objective': 'multiclass',  # 3クラス分類
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'max_depth': 8,
            'num_boost_round': 1000,
            'early_stopping_rounds': 100,
            'random_state': 42,
            'verbose': -1,
            'class_weight': 'balanced'
        }
    
    def prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        データの準備（読み込み、特徴量計算、目的変数計算）
        
        Returns:
            features, targets, labels
        """
        print("=== データ準備開始 ===")
        
        # 1. 相場データ読み込み
        data = self.data_pipeline.load_market_data()
        
        # 2. 特徴量計算
        print("\n特徴量計算中...")
        features = self.feature_engineering.calculate_all_features(data)
        self.feature_names = features.columns.tolist()
        
        # 3. 目的変数計算
        print("\n目的変数計算中...")
        targets = self.target_calculation.calculate_atr_normalized_target(data)
        
        # 4. クラスラベル作成
        labels = self.target_calculation.create_target_labels(targets)
        
        print(f"\nデータ準備完了:")
        print(f"  特徴量: {features.shape}")
        print(f"  目的変数: {len(targets)} (ユニーク値: {np.unique(targets)})")
        print(f"  クラスラベル: {len(labels)} (ユニーク値: {np.unique(labels)})")
        
        return features, targets, labels
    
    def time_series_split_validation(self, 
                                   features: pd.DataFrame, 
                                   labels: np.ndarray, 
                                   n_splits: int = 5) -> Dict[str, Any]:
        """
        時系列分割交差検証
        
        Args:
            features: 特徴量データフレーム
            labels: クラスラベル
            n_splits: 分割数
        
        Returns:
            交差検証結果
        """
        print(f"\n=== 時系列交差検証開始 (分割数: {n_splits}) ===")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        cv_reports = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # データ分割
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # クラス分布チェック
            train_dist = np.bincount(y_train)
            val_dist = np.bincount(y_val)
            print(f"  訓練データクラス分布: {train_dist}")
            print(f"  検証データクラス分布: {val_dist}")
            
            # モデル訓練
            model = self._train_single_model(X_train, y_train, X_val, y_val)
            
            # 予測・評価
            val_pred_proba = model.predict_proba(X_val)
            val_pred_class = np.argmax(val_pred_proba, axis=1)
            
            accuracy = accuracy_score(y_val, val_pred_class)
            cv_scores.append(accuracy)
            
            # 詳細レポート
            report = classification_report(y_val, val_pred_class, output_dict=True)
            cv_reports.append(report)
            
            print(f"  Fold {fold + 1} Accuracy: {accuracy:.4f}")
        
        # 交差検証結果のまとめ
        cv_result = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'fold_scores': cv_scores,
            'fold_reports': cv_reports
        }
        
        print(f"\n交差検証結果:")
        print(f"  平均精度: {cv_result['mean_accuracy']:.4f} ± {cv_result['std_accuracy']:.4f}")
        
        return cv_result
    
    def train_final_model(self, 
                         features: pd.DataFrame, 
                         labels: np.ndarray,
                         train_ratio: float = 0.85) -> lgb.LGBMClassifier:
        """
        最終モデルの訓練
        
        Args:
            features: 特徴量データフレーム
            labels: クラスラベル
            train_ratio: 訓練データの割合
        
        Returns:
            訓練済みモデル
        """
        print(f"\n=== 最終モデル訓練開始 ===")
        
        # データ分割
        split_idx = int(len(features) * train_ratio)
        X_train = features.iloc[:split_idx]
        X_val = features.iloc[split_idx:]
        y_train = labels[:split_idx]
        y_val = labels[split_idx:]
        
        print(f"訓練データ: {X_train.shape}")
        print(f"検証データ: {X_val.shape}")
        
        # 特徴量名を保存
        self.feature_names = features.columns.tolist()
        
        # モデル訓練
        self.model = self._train_single_model(X_train, y_train, X_val, y_val)
        
        # 最終評価
        train_pred_proba = self.model.predict_proba(X_train)
        val_pred_proba = self.model.predict_proba(X_val)
        
        train_pred_class = np.argmax(train_pred_proba, axis=1)
        val_pred_class = np.argmax(val_pred_proba, axis=1)
        
        train_accuracy = accuracy_score(y_train, train_pred_class)
        val_accuracy = accuracy_score(y_val, val_pred_class)
        
        print(f"\n最終モデル性能:")
        print(f"  訓練精度: {train_accuracy:.4f}")
        print(f"  検証精度: {val_accuracy:.4f}")
        
        # 詳細レポート
        print(f"\n詳細評価レポート:")
        print(classification_report(y_val, val_pred_class, 
                                  target_names=['失敗・中立', '買い成功', '売り成功']))
        
        return self.model
    
    def _train_single_model(self, 
                          X_train: pd.DataFrame, 
                          y_train: np.ndarray,
                          X_val: pd.DataFrame, 
                          y_val: np.ndarray) -> lgb.LGBMClassifier:
        """
        単一モデルの訓練
        
        Args:
            X_train, y_train: 訓練データ
            X_val, y_val: 検証データ
        
        Returns:
            訓練済みモデル
        """
        # LightGBMモデル初期化
        model = lgb.LGBMClassifier(**self.lgb_params)
        
        # 訓練実行
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(self.lgb_params['early_stopping_rounds']), lgb.log_evaluation(0)]
        )
        
        return model
    
    def walk_forward_analysis(self, 
                            features: pd.DataFrame, 
                            labels: np.ndarray,
                            train_ratio: float = 0.5) -> Dict[str, Any]:
        """
        シンプルな50-50分割ウォークフォワード分析
        
        Args:
            features: 特徴量データフレーム
            labels: クラスラベル
            train_ratio: 訓練データの割合（デフォルト: 0.5）
        
        Returns:
            ウォークフォワード結果
        """
        print(f"\n=== シンプルウォークフォワード分析開始 ===")
        
        total_len = len(features)
        train_size = int(total_len * train_ratio)
        
        print(f"総データ数: {total_len}")
        print(f"訓練データ: 0 - {train_size}")
        print(f"テストデータ: {train_size} - {total_len}")
        
        # データ分割
        X_train = features.iloc[:train_size]
        X_test = features.iloc[train_size:]
        y_train = labels[:train_size]
        y_test = labels[train_size:]
        
        # 検証データは訓練データの最後20%
        val_split = int(len(X_train) * 0.8)
        X_train_sub = X_train.iloc[:val_split]
        X_val_sub = X_train.iloc[val_split:]
        y_train_sub = y_train[:val_split]
        y_val_sub = y_train[val_split:]
        
        print(f"  訓練データ分割: {len(X_train_sub)} / {len(X_val_sub)}")
        
        # モデル訓練
        model = self._train_single_model(X_train_sub, y_train_sub, X_val_sub, y_val_sub)
        
        # テストデータで評価
        test_pred_proba = model.predict_proba(X_test)
        test_pred_class = np.argmax(test_pred_proba, axis=1)
        test_accuracy = accuracy_score(y_test, test_pred_class)
        
        print(f"テスト精度: {test_accuracy:.4f}")
        
        # 結果保存
        result = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_accuracy': test_accuracy,
            'class_distribution_train': np.bincount(y_train).tolist(),
            'class_distribution_test': np.bincount(y_test).tolist(),
            'confusion_matrix': confusion_matrix(y_test, test_pred_class).tolist(),
            'model': model
        }
        
        return result
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        特徴量重要度分析
        
        Returns:
            特徴量重要度の分析結果
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        print("\n=== 特徴量重要度分析 ===")
        
        # LightGBM特徴量重要度
        importance_gain = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_gain
        }).sort_values('importance', ascending=False)
        
        print(f"Top 10 重要特徴量:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'feature_importance': importance_df.to_dict('records'),
            'top_features': importance_df.head(20)['feature'].tolist()
        }
    
    def save_model(self, filepath: str) -> None:
        """
        モデルを保存
        
        Args:
            filepath: 保存先パス
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        # モデル保存
        model_path = filepath
        joblib.dump(self.model, model_path)
        
        # メタデータ保存
        metadata = {
            'model_type': 'LightGBM',
            'model_params': self.lgb_params,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names) if self.feature_names is not None else 0,
            'training_timestamp': datetime.now().isoformat(),
            'target_info': self.target_calculation.get_target_info()
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"モデル保存完了:")
        print(f"  モデル: {model_path}")
        print(f"  メタデータ: {metadata_path}")
    
    def load_model(self, filepath: str) -> lgb.LGBMClassifier:
        """
        モデルを読み込み
        
        Args:
            filepath: モデルファイルパス
        
        Returns:
            読み込んだモデル
        """
        self.model = joblib.load(filepath)
        
        # メタデータ読み込み
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.feature_names = metadata.get('feature_names', [])
            print(f"モデル読み込み完了: {len(self.feature_names)}特徴量")
        
        return self.model


def main():
    """メイン実行関数"""
    # モデル訓練器初期化
    trainer = TrendFollowModelTraining()
    
    # データ準備
    features, targets, labels = trainer.prepare_data()
    
    # 時系列交差検証
    cv_result = trainer.time_series_split_validation(features, labels)
    
    # 最終モデル訓練
    model = trainer.train_final_model(features, labels)
    
    # 特徴量重要度分析
    importance_result = trainer.analyze_feature_importance()
    
    # モデル保存
    model_path = f"ml/trend_follow_v1/models/lightgbm_trend_follow_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    trainer.save_model(model_path)
    
    print("\n=== 訓練完了 ===")


if __name__ == "__main__":
    main()