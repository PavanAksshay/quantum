"""
Pydantic schemas for Prediction requests and responses.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: Optional[str] = Field(default="", description="Single text input string (for SMS or combined text)")
    subject: Optional[str] = Field(default=None, description="Email subject line")
    body: Optional[str] = Field(default=None, description="Email body text")
    representation: Optional[str] = Field(default="tfidf", description="Representation ID: tfidf, roberta, minilm, mpnet, fasttext")
    dimension: Optional[int] = Field(default=8, description="Target projected representation dimension (qubits)")
    dataset: Optional[str] = Field(default="MeAJOR", description="Reference benchmark dataset: MeAJOR, CEAS_08, SMS")


class StatevectorRequest(BaseModel):
    text: str
    representation: Optional[str] = "tfidf"
    dimension: Optional[int] = 8
