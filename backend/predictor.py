"""
RECOV.AI - Recovery Prediction Engine
=====================================
This module provides the core ML prediction functionality for debt recovery. 
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import os

class RecoveryPredictor:
    """
    Main prediction engine for RECOV.AI. 
    Loads trained XGBoost models and provides recovery predictions.
    """
    
    def __init__(self, model_path: str = 'backend/models/recovery_model.pkl'):
        """Initialize predictor and load models"""
        self.model_path = Path(model_path)
        self.models = None
        self.feature_names = None
        self.explainer = None
        
        # Find model file (handle different working directories)
        if not self.model_path.exists():
            alt_paths = [
                Path("models/recovery_model.pkl"),
                Path("backend/models/recovery_model.pkl"),
                Path("../backend/models/recovery_model.pkl"),
                Path(__file__).parent / "models" / "recovery_model.pkl"
            ]
            for p in alt_paths:
                if p.exists():
                    self.model_path = p
                    break
        
        # Load models
        self.load_models()
        
        # Try to load SHAP explainer
        try:
            from backend.shap_explainer import ExplainabilityEngine
            self.explainer = ExplainabilityEngine(str(self.model_path))
            print("✅ SHAP Explainer loaded")
        except Exception as e:
            print(f"⚠️ SHAP Explainer unavailable: {e}")
            self.explainer = None

    def load_models(self):
        """Load pickled models and metadata"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        print(f"📂 Loading model from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            pkg = pickle.load(f)
            self.models = pkg['models']
            self.feature_names = pkg['feature_names']
        
        print("✅ Models loaded successfully")

    def prepare_features(self, account: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform raw account data into model features.
        Handles missing columns gracefully.
        """
        # Create DataFrame from account dict
        df = pd.DataFrame([account])
        
        # Feature Engineering
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
        
        # Initialize feature dataframe with zeros
        X_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # Map numerical features
        numerical_features = [
            'amount_log', 'days_overdue', 'payment_history_score',
            'shipment_volume_change_30d', 'shipment_volume_30d',
            'express_ratio', 'destination_diversity'
        ]
        
        for col in numerical_features:
            if col in df.columns and col in X_df.columns:
                X_df[col] = df[col].iloc[0]
            elif col == 'amount_log' and 'amount_log' in X_df.columns:
                X_df[col] = df[col].iloc[0]
        
        # Handle boolean features
        boolean_features = ['email_opened', 'dispute_flag']
        for col in boolean_features:
            if col in df.columns and col in X_df.columns:
                val = df[col].iloc[0]
                # Convert string/bool to int
                if isinstance(val, str):
                    X_df[col] = 1 if val.upper() in ['TRUE', '1', 'YES'] else 0
                else: 
                    X_df[col] = int(bool(val))
        
        # Handle categorical features (one-hot encoding)
        for col in ['industry', 'region']: 
            if col in df.columns: 
                value = df[col].iloc[0]
                encoded_col = f"{col}_{value}"
                if encoded_col in X_df.columns:
                    X_df[encoded_col] = 1
        
        return X_df

    def match_dca(self, account: Dict[str, Any], recovery_prob: float) -> Dict[str, Any]: 
        """
        Match account to optimal DCA based on characteristics.
        This is a rule-based matching algorithm.
        """
        amount = account.get('amount', 0)
        industry = account.get('industry', 'Other')
        
        # DCA Matching Logic
        if amount > 1000000 and recovery_prob > 0.7:
            return {
                "name": "Premium Recovery Services",
                "specialization": "High-value B2B accounts",
                "reasoning": "Specialized in large corporate accounts with high recovery potential"
            }
        elif industry in ['Technology', 'Tech'] and recovery_prob > 0.6:
            return {
                "name": "TechCollect Pro",
                "specialization": "Technology sector",
                "reasoning": "Expert in tech industry payment cycles and negotiations"
            }
        elif recovery_prob < 0.4: 
            return {
                "name": "Recovery Specialists Inc",
                "specialization": "Challenging cases",
                "reasoning": "Experienced in difficult recovery scenarios with legal support"
            }
        else: 
            return {
                "name": "Standard Recovery Partners",
                "specialization": "General collections",
                "reasoning": "Reliable performance across all account types"
            }

    def calculate_risk_level(self, recovery_prob: float) -> str:
        """Determine risk level based on recovery probability"""
        if recovery_prob >= 0.75:
            return "Low"
        elif recovery_prob >= 0.50:
            return "Medium"
        elif recovery_prob >= 0.25:
            return "High"
        else:
            return "Very High"

    def predict_recovery(self, account: Dict[str, Any]) -> Dict[str, Any]: 
        """
        Main prediction method. 
        Returns complete prediction with all metadata.
        """
        try:
            # Prepare features
            X_df = self.prepare_features(account)
            
            # Get predictions from all models
            classifier = self.models['classifier']
            regressor_days = self.models.get('regressor_days')
            regressor_pct = self.models.get('regressor_pct')
            
            # Recovery probability
            recovery_prob = float(classifier.predict_proba(X_df)[0][1])
            
            # Days to recovery
            if regressor_days:
                expected_days = int(regressor_days.predict(X_df)[0])
                expected_days = max(5, min(expected_days, 90))  # Clamp between 5-90 days
            else:
                # Fallback calculation
                expected_days = int(30 + (1 - recovery_prob) * 30)
            
            # Recovery percentage
            if regressor_pct:
                recovery_pct = float(regressor_pct.predict(X_df)[0])
                recovery_pct = max(0.5, min(recovery_pct, 1.0))  # Clamp between 50-100%
            else: 
                recovery_pct = recovery_prob * 0.85  # Fallback
            
            # Calculate Recovery Velocity Score
            # Formula: (Probability × Recovery%) / Days × 1000
            recovery_velocity = (recovery_prob * recovery_pct / expected_days) * 1000
            
            # Match DCA
            dca = self.match_dca(account, recovery_prob)
            
            # Risk level
            risk_level = self.calculate_risk_level(recovery_prob)
            
            # Get SHAP explanations
            top_factors = []
            if self.explainer:
                try:
                    explanation = self.explainer.explain_prediction(X_df)
                    top_factors = explanation.get('top_factors', [])[:5]
                except Exception as e:
                    print(f"⚠️ SHAP explanation failed: {e}")
            
            # Fallback: Use feature importances if SHAP fails
            if not top_factors: 
                try:
                    importances = classifier.feature_importances_
                    top_indices = np.argsort(importances)[-5:][::-1]
                    top_factors = [
                        {
                            "feature": self.feature_names[i],
                            "impact": float(importances[i]),
                            "direction": "positive" if X_df.iloc[0, i] > 0 else "neutral"
                        }
                        for i in top_indices
                    ]
                except: 
                    top_factors = [
                        {"feature": "payment_history_score", "impact": 0.3, "direction": "positive"},
                        {"feature": "shipment_volume_change_30d", "impact": 0.25, "direction": "positive"},
                        {"feature": "days_overdue", "impact": -0.2, "direction": "negative"}
                    ]
            
            # Build response
            response = {
                "account_id": account.get('account_id', 'UNKNOWN'),
                "company_name": account.get('company_name', 'Unknown Company'),
                "recovery_probability": round(recovery_prob, 4),
                "recovery_percentage": round(recovery_pct, 4),
                "expected_days": expected_days,
                "recovery_velocity_score": round(recovery_velocity, 2),
                "risk_level": risk_level,
                "recommended_dca": dca,
                "top_factors": top_factors,
                "prediction_timestamp": pd.Timestamp.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error response
            return {
                "account_id": account.get('account_id', 'ERROR'),
                "company_name": account.get('company_name', 'Error'),
                "recovery_probability": 0.0,
                "recovery_percentage": 0.0,
                "expected_days": 30,
                "recovery_velocity_score": 0.0,
                "risk_level": "Unknown",
                "recommended_dca": {
                    "name": "Error",
                    "specialization": "N/A",
                    "reasoning": f"Prediction failed: {str(e)}"
                },
                "top_factors": [],
                "prediction_timestamp": pd.Timestamp.now().isoformat(),
                "error": str(e)
            }