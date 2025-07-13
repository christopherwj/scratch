import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class MLTrader:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, data: pd.DataFrame, feature_cols: list, target_col: str, test_size: float = 0.2):
        """
        Prepares data for training with proper time series split.
        """
        # Remove rows with NaN in target
        clean_data = data.dropna(subset=[target_col])
        
        X = clean_data[feature_cols].copy()
        y = clean_data[target_col].copy()
        
        # Time series split (no random shuffling to avoid look-ahead bias)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.scalers['features'] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Trains XGBoost model with GPU acceleration.
        """
        print("Training XGBoost...")
        
        # XGBoost parameters optimized for GPU
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'gpu_id': 0 if self.use_gpu else None,
            'early_stopping_rounds': 50,
            'eval_metric': 'mlogloss'
        }
        
        # Convert target to 0, 1, 2 for XGBoost
        y_train_xgb = y_train + 1  # Convert -1,0,1 to 0,1,2
        y_test_xgb = y_test + 1
        
        model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train_xgb,
            eval_set=[(X_test, y_test_xgb)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_original = y_pred - 1  # Convert back to -1,0,1
        
        accuracy = accuracy_score(y_test, y_pred_original)
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = dict(zip(X_train.columns, model.feature_importances_))
        
        return accuracy
    
    def train_catboost(self, X_train, y_train, X_test, y_test):
        """
        Trains CatBoost model with GPU acceleration.
        """
        print("Training CatBoost...")
        
        # Convert target to 0, 1, 2 for CatBoost
        y_train_cb = y_train + 1
        y_test_cb = y_test + 1
        
        model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            task_type='GPU' if self.use_gpu else 'CPU',
            gpu_device_id=0 if self.use_gpu else None,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Train with validation set
        model.fit(
            X_train, y_train_cb,
            eval_set=(X_test, y_test_cb),
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_original = y_pred - 1  # Convert back to -1,0,1
        
        accuracy = accuracy_score(y_test, y_pred_original)
        print(f"CatBoost Accuracy: {accuracy:.4f}")
        
        self.models['catboost'] = model
        self.feature_importance['catboost'] = dict(zip(X_train.columns, model.feature_importances_))
        
        return accuracy
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Trains Random Forest model.
        """
        print("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = dict(zip(X_train.columns, model.feature_importances_))
        
        return accuracy
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Creates an ensemble of all trained models.
        """
        print("Creating ensemble...")
        
        predictions = {}
        for name, model in self.models.items():
            if name in ['xgboost', 'catboost']:
                # Convert predictions back to -1,0,1
                pred = model.predict(X_test) - 1
            else:
                pred = model.predict(X_test)
            predictions[name] = pred
        
        # Simple voting ensemble
        ensemble_pred = np.zeros(len(y_test))
        for pred in predictions.values():
            ensemble_pred += pred
        
        # Convert to discrete signals
        ensemble_pred = np.sign(ensemble_pred)
        
        accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        
        return accuracy, ensemble_pred
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Trains all models and finds the best one.
        """
        print("="*60)
        print("TRAINING ML MODELS")
        print("="*60)
        
        results = {}
        
        # Train individual models
        results['xgboost'] = self.train_xgboost(X_train, y_train, X_test, y_test)
        results['catboost'] = self.train_catboost(X_train, y_train, X_test, y_test)
        results['random_forest'] = self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train ensemble
        ensemble_acc, ensemble_pred = self.train_ensemble(X_train, y_train, X_test, y_test)
        results['ensemble'] = ensemble_acc
        
        # Find best model
        self.best_model = max(results, key=results.get)
        self.best_score = results[self.best_model]
        
        print(f"\nBest Model: {self.best_model} (Accuracy: {self.best_score:.4f})")
        
        return results
    
    def predict(self, X):
        """
        Makes predictions using the best model.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        if self.best_model == 'ensemble':
            # Ensemble prediction
            predictions = {}
            for name, model in self.models.items():
                if name in ['xgboost', 'catboost']:
                    pred = model.predict(X_scaled) - 1
                else:
                    pred = model.predict(X_scaled)
                predictions[name] = pred
            
            ensemble_pred = np.zeros(len(X_scaled))
            for pred in predictions.values():
                ensemble_pred += pred
            
            return np.sign(ensemble_pred)
        else:
            model = self.models[self.best_model]
            if self.best_model in ['xgboost', 'catboost']:
                return model.predict(X_scaled) - 1
            else:
                return model.predict(X_scaled)
    
    def get_feature_importance(self, top_n=20):
        """
        Returns feature importance from the best model.
        """
        if self.best_model not in self.feature_importance:
            return None
        
        importance = self.feature_importance[self.best_model]
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_importance[:top_n]
    
    def print_model_summary(self):
        """
        Prints a summary of all trained models.
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"{name.upper()}: Available")
        
        print(f"\nBest Model: {self.best_model}")
        print(f"Best Accuracy: {self.best_score:.4f}")
        
        # Feature importance
        if self.best_model in self.feature_importance:
            print(f"\nTop 10 Features ({self.best_model}):")
            top_features = self.get_feature_importance(10)
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature}: {importance:.4f}")

if __name__ == '__main__':
    # Test the ML trainer
    from src.ml_features import FeatureEngineer
    from src.data_loader import fetch_data
    
    print("Testing ML Trader...")
    
    # Load data
    data = fetch_data('AAPL', '2020-01-01', '2023-12-31')
    if data is not None:
        # Engineer features
        engineer = FeatureEngineer()
        features_df = engineer.create_features(data)
        targets_df = engineer.create_targets(features_df)
        
        # Prepare for ML
        feature_cols = engineer.get_feature_importance_names()
        
        # Initialize trainer
        trainer = MLTrader(use_gpu=True)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            targets_df, feature_cols, 'target_signal'
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Train models
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Print summary
        trainer.print_model_summary() 