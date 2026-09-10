"""
Pydantic schemas for Research metrics, tables, and diagnostics.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class TableMetadata(BaseModel):
    id: str
    title: str
    file: str
    status: str
    type: str


class TableContentResponse(BaseModel):
    id: str
    title: str
    file_name: str
    status: str
    type: str
    columns: List[str]
    rows: List[Dict[str, Any]]
