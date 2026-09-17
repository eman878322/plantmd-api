from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Environment(BaseModel):
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100)
    rainfall_mm_24h: Optional[float] = Field(default=None, ge=0)

class AnalyzeRequest(BaseModel):
    crop: str
    disease: Optional[str] = None
    severity_percent: Optional[float] = Field(default=None, ge=0, le=100)
    stage: Optional[str] = None
    environment: Optional[Environment] = None
    farmer_notes: Optional[str] = None

class Diagnosis(BaseModel):
    disease: str
    confidence: float
    disease_type: Optional[str] = None
    crop: Optional[str] = None

class ProgressionPoint(BaseModel):
    days: int
    severity_percent: float
    basis: str

class AnalyzeResponse(BaseModel):
    request_id: str
    report: Dict[str, Any]
    retrieved_sources: List[Dict[str, Any]]
    warnings: List[str] = []
