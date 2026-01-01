"""
RECOV.AI - Pydantic Data Models
================================
Type-safe data structures for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ============================================================================
# INPUT MODELS
# ============================================================================

class AccountData(BaseModel):
    """Input model for account data"""
    account_id: str
    company_name: str = "Unknown"
    amount: float
    days_overdue: int
    payment_history_score: float
    shipment_volume_change_30d: float
    
    # Optional fields with defaults
    shipment_volume_30d: Optional[int] = 0
    express_ratio: Optional[float] = 0.0
    destination_diversity: Optional[int] = 0
    industry: Optional[str] = "Other"
    region: Optional[str] = "Other"
    email_opened: Optional[bool] = False
    dispute_flag: Optional[bool] = False
    
    class Config:
        extra = "ignore"  # Ignore extra fields

# ============================================================================
# OUTPUT MODELS
# ============================================================================

class DCARecommendation(BaseModel):
    """DCA recommendation details"""
    name: str
    specialization: str
    reasoning: str = "Recommended based on account profile"

class TopFactor(BaseModel):
    """Feature importance factor"""
    feature: str
    impact: float
    direction: str

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    account_id: str
    company_name: str
    recovery_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of recovery (0-1)")
    recovery_percentage: float = Field(..., ge=0.0, le=1.0, description="Expected recovery percentage")
    expected_days: int = Field(..., ge=1, le=180, description="Expected days to recovery")
    recovery_velocity_score: float = Field(..., ge=0.0, description="Recovery velocity metric")
    risk_level: str = Field(..., description="Risk category: Low/Medium/High/Very High")
    recommended_dca: DCARecommendation
    top_factors: List[TopFactor]
    prediction_timestamp: str
    error: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    """Batch analysis response"""
    total_accounts: int
    predictions: List[PredictionResponse]
    summary: dict