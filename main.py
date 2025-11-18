import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from database import db, create_document, get_documents
from pydantic import BaseModel, EmailStr
from schemas import User, Product, Sale, SaleItem, Settings

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

app = FastAPI(title="Supermarket POS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility helpers
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None


class ProductCreate(BaseModel):
    barcode: str
    name: str
    price: float
    cost: float = 0
    stock: int = 0
    category: Optional[str] = None
    tax_rate: float = 0.0


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    cost: Optional[float] = None
    stock: Optional[int] = None
    category: Optional[str] = None
    tax_rate: Optional[float] = None
    is_active: Optional[bool] = None


class StockAdjust(BaseModel):
    barcode: str
    delta: int


class SaleRequest(BaseModel):
    items: List[SaleItem]
    paid_amount: float
    payment_method: str = "cash"


class AuthRequest(BaseModel):
    email: EmailStr
    password: str


# Auth helpers

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_docs = get_documents("user", {"email": email}, limit=1)
    if not user_docs:
        raise credentials_exception
    return user_docs[0]


@app.get("/")
def read_root():
    return {"message": "Supermarket POS API"}


@app.get("/test")
def test_database():
    info = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            info["database"] = "✅ Connected & Working"
            info["database_url"] = "✅ Set"
            info["database_name"] = db.name
            info["connection_status"] = "Connected"
            info["collections"] = db.list_collection_names()
    except Exception as e:
        info["database"] = f"⚠️ Error: {str(e)[:80]}"
    return info


# Helper to accept either JSON or form for legacy compatibility
async def parse_auth_request(request: Request) -> AuthRequest:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        return AuthRequest(**data)
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        email = (form.get("username") or form.get("email") or "").lower()
        password = form.get("password") or ""
        return AuthRequest(email=email, password=password)
    else:
        data = await request.json()
        return AuthRequest(**data)


# Auth routes
@app.post("/auth/register", response_model=Token)
async def register(request: Request):
    auth = await parse_auth_request(request)

    existing = get_documents("user", {"email": auth.email}, limit=1)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=auth.email, name=auth.email.split("@")[0], password_hash=get_password_hash(auth.password))
    create_document("user", user)

    token = create_access_token({"sub": user.email})
    return Token(access_token=token)


@app.post("/auth/token", response_model=Token)
async def login(request: Request):
    auth = await parse_auth_request(request)

    user_docs = get_documents("user", {"email": auth.email}, limit=1)
    if not user_docs:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = user_docs[0]
    if not verify_password(auth.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": auth.email})
    return Token(access_token=token)


# Products
@app.post("/products")
def create_product(product: ProductCreate, _: dict = Depends(get_current_user)):
    existing = get_documents("product", {"barcode": product.barcode}, limit=1)
    if existing:
        raise HTTPException(status_code=400, detail="Barcode already exists")
    create_document("product", Product(**product.model_dump()))
    return {"status": "ok"}


@app.put("/products/{barcode}")
def update_product(barcode: str, update: ProductUpdate, _: dict = Depends(get_current_user)):
    doc = db.product.find_one({"barcode": barcode})
    if not doc:
        raise HTTPException(status_code=404, detail="Product not found")
    update_dict = {k: v for k, v in update.model_dump(exclude_none=True).items()}
    if not update_dict:
        return {"status": "ok"}
    db.product.update_one({"barcode": barcode}, {"$set": update_dict, "$currentDate": {"updated_at": True}})
    return {"status": "ok"}


@app.delete("/products/{barcode}")
def delete_product(barcode: str, _: dict = Depends(get_current_user)):
    res = db.product.delete_one({"barcode": barcode})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"status": "ok"}


@app.post("/products/adjust-stock")
def adjust_stock(req: StockAdjust, _: dict = Depends(get_current_user)):
    res = db.product.update_one({"barcode": req.barcode}, {"$inc": {"stock": req.delta}})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"status": "ok"}


@app.get("/products/by-barcode/{barcode}")
def get_product_by_barcode(barcode: str, _: dict = Depends(get_current_user)):
    docs = get_documents("product", {"barcode": barcode}, limit=1)
    if not docs:
        raise HTTPException(status_code=404, detail="Product not found")
    doc = docs[0]
    doc["_id"] = str(doc.get("_id"))
    return doc


@app.get("/products", response_model=list)
def list_products(_: dict = Depends(get_current_user)):
    docs = get_documents("product", {})
    for d in docs:
        d["_id"] = str(d.get("_id"))
    return docs


# Sales
@app.post("/sales")
def create_sale(req: SaleRequest, user: dict = Depends(get_current_user)):
    # Compute totals server-side
    subtotal = sum(item.price * item.quantity for item in req.items)
    tax_total = sum((item.price * item.quantity) * item.tax_rate for item in req.items)
    total = subtotal + tax_total

    sale = Sale(
        user_id=str(user.get("_id")),
        items=req.items,
        subtotal=subtotal,
        tax_total=tax_total,
        total=total,
        paid_amount=req.paid_amount,
        change=max(0.0, req.paid_amount - total),
        payment_method=req.payment_method,
        created_at=datetime.now(timezone.utc)
    )
    create_document("sale", sale)

    # Decrement stock
    try:
        for item in req.items:
            db.product.update_one({"barcode": item.barcode}, {"$inc": {"stock": -item.quantity}})
    except Exception:
        pass

    return {"status": "ok", "total": total}


@app.get("/sales/summary")
def sales_summary(_: dict = Depends(get_current_user)):
    # Simple aggregation summaries
    try:
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)

        pipeline = [
            {"$match": {"created_at": {"$gte": start, "$lte": end}}},
            {"$group": {"_id": None, "total": {"$sum": "$total"}, "txns": {"$count": {}}, "items": {"$sum": {"$size": "$items"}}}}
        ]
        res = list(db.sale.aggregate(pipeline))
        if res:
            s = res[0]
            return {"today_total": s.get("total", 0), "transactions": s.get("txns", 0), "items_sold": s.get("items", 0)}
    except Exception:
        pass
    return {"today_total": 0, "transactions": 0, "items_sold": 0}


# Advanced reports
@app.get("/sales/range-summary")
def range_summary(start: Optional[str] = None, end: Optional[str] = None, _: dict = Depends(get_current_user)):
    try:
        if start:
            start_dt = datetime.fromisoformat(start)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=7)
        if end:
            end_dt = datetime.fromisoformat(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)
        pipeline = [
            {"$match": {"created_at": {"$gte": start_dt, "$lte": end_dt}}},
            {"$unwind": "$items"},
            {"$group": {
                "_id": None,
                "revenue": {"$sum": "$total"},
                "transactions": {"$addToSet": "$_id"},
                "items_sold": {"$sum": "$items.quantity"},
            }},
            {"$project": {"_id": 0, "revenue": 1, "transactions": {"$size": "$transactions"}, "items_sold": 1}}
        ]
        res = list(db.sale.aggregate(pipeline))
        if res:
            return res[0]
    except Exception:
        pass
    return {"revenue": 0, "transactions": 0, "items_sold": 0}


@app.get("/sales/top-products")
def top_products(days: int = 30, limit: int = 10, _: dict = Depends(get_current_user)):
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"created_at": {"$gte": start_dt}}},
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.barcode", "name": {"$first": "$items.name"}, "qty": {"$sum": "$items.quantity"}, "revenue": {"$sum": {"$multiply": ["$items.price", "$items.quantity"]}}}},
            {"$sort": {"qty": -1}},
            {"$limit": limit},
            {"$project": {"barcode": "$_id", "_id": 0, "name": 1, "qty": 1, "revenue": 1}}
        ]
        return list(db.sale.aggregate(pipeline))
    except Exception:
        return []


@app.get("/sales/export")
def export_sales(start: Optional[str] = None, end: Optional[str] = None, _: dict = Depends(get_current_user)):
    import csv
    from io import StringIO
    try:
        if start:
            start_dt = datetime.fromisoformat(start)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(days=7)
        if end:
            end_dt = datetime.fromisoformat(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)

        cursor = db.sale.find({"created_at": {"$gte": start_dt, "$lte": end_dt}})
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["_id", "created_at", "subtotal", "tax_total", "total", "paid_amount", "change", "payment_method", "items_count"])
        for s in cursor:
            writer.writerow([
                str(s.get("_id")), s.get("created_at"), s.get("subtotal"), s.get("tax_total"), s.get("total"), s.get("paid_amount"), s.get("change"), s.get("payment_method"), len(s.get("items", []))
            ])
        return output.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Settings
@app.get("/settings")
def get_settings(_: dict = Depends(get_current_user)):
    doc = db.settings.find_one({})
    if not doc:
        default = Settings()
        return default.model_dump()
    doc.pop("_id", None)
    return doc


@app.put("/settings")
def update_settings(s: Settings, _: dict = Depends(get_current_user)):
    db.settings.update_one({}, {"$set": s.model_dump(), "$currentDate": {"updated_at": True}}, upsert=True)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
