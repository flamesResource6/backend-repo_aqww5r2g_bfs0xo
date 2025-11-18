"""
Database Schemas for Supermarket POS

Each Pydantic model represents a collection in MongoDB. Collection name is the lowercase of the class name.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class User(BaseModel):
    email: EmailStr
    name: str
    password_hash: str
    role: str = Field("cashier", description="user role: admin, manager, cashier")
    is_active: bool = True


class Product(BaseModel):
    barcode: str = Field(..., description="EAN/UPC/Code128 etc.")
    name: str
    price: float = Field(..., ge=0)
    cost: float = Field(0, ge=0)
    stock: int = Field(0, ge=0)
    category: Optional[str] = None
    tax_rate: float = Field(0.0, ge=0)
    is_active: bool = True


class SaleItem(BaseModel):
    barcode: str
    name: str
    price: float
    quantity: int
    tax_rate: float = 0.0


class Sale(BaseModel):
    user_id: str
    items: List[SaleItem]
    subtotal: float
    tax_total: float
    total: float
    paid_amount: float
    change: float
    payment_method: str = "cash"  # cash, card, mobile
    created_at: Optional[datetime] = None
