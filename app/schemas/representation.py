"""
Pydantic schemas for Representation Lab.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class RepresentationMetadata(BaseModel):
    id: str
    name: str
    representation_type: str
    is_sparse: bool
    original_dimension: int
    target_dimension: int
    status: str
    status_message: str
    description: str
    preprocessing_details: str
    is_canonical: bool
    is_fitted: bool
