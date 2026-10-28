"""
Authentication REST Router (`/api/v1/auth`) and User Sessions (`/api/v1/sessions`).
"""

import uuid
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select

from app.auth.security import (
    check_login_rate_limit,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.config import settings
from app.database.db import AsyncSessionLocal, UserModel, ensure_db_ready, list_user_sessions
from app.schemas.clinical_schemas import (
    GoogleAuthRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserSignupRequest,
)

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
session_router = APIRouter(prefix="/api/v1/sessions", tags=["User Sessions"])


@auth_router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup_user(payload: UserSignupRequest):
    await ensure_db_ready()
    async with AsyncSessionLocal() as db:
        # Check duplicate email
        existing = await db.execute(select(UserModel).where(UserModel.email == payload.email.lower().strip()))
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="البريد الإلكتروني مسجل بالفعل. يرجى تسجيل الدخول أو استخدام بريد آخر.",
            )

        hashed = hash_password(payload.password)
        new_user = UserModel(
            email=payload.email.lower().strip(),
            phone=payload.phone.strip() if payload.phone else None,
            full_name=payload.full_name.strip(),
            password_hash=hashed,
            role=payload.role.value,
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        access_token = create_access_token(new_user.id, new_user.email, new_user.role)
        refresh_token = create_refresh_token(new_user.id)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MIN * 60,
        )


@auth_router.post("/login", response_model=TokenResponse)
async def login_user(payload: UserLoginRequest, request: Request):
    check_login_rate_limit(request)
    await ensure_db_ready()

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(UserModel).where(UserModel.email == payload.email.lower().strip()))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="هذا الحساب غير مسجل بعد في قاعدة البيانات. يمكنك الضغط على 'إنشاء حساب جديد' للتسجيل فوراً.",
            )
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="كلمة المرور غير صحيحة، يرجى إعادة المحاولة.",
            )

        access_token = create_access_token(user.id, user.email, user.role)
        refresh_token = create_refresh_token(user.id)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MIN * 60,
        )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(payload: RefreshTokenRequest):
    await ensure_db_ready()
    decoded = decode_token(payload.refresh_token)
    if decoded.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user_id = int(decoded.get("sub", 0))
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(UserModel).where(UserModel.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        new_access = create_access_token(user.id, user.email, user.role)
        new_refresh = create_refresh_token(user.id)
        return TokenResponse(
            access_token=new_access,
            refresh_token=new_refresh,
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MIN * 60,
        )


@auth_router.get("/google/config")
async def get_google_auth_config():
    """Returns whether Google Sign-in is configured and its public Client ID."""
    return {
        "client_id": settings.GOOGLE_CLIENT_ID or "",
        "enabled": bool(settings.GOOGLE_CLIENT_ID),
    }


@auth_router.post("/google", response_model=TokenResponse)
async def authenticate_with_google(payload: GoogleAuthRequest):
    """
    Authenticates or creates a user using Google OAuth / GIS token.
    Accepts signed Google ID token (credential) or direct verified profile.
    """
    await ensure_db_ready()

    email: str = ""
    full_name: str = ""

    if payload.credential:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={payload.credential}")
                if res.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="رمز اعتماد Google غير صالح أو منتهي الصلاحية.",
                    )
                info = res.json()
                email = info.get("email", "").lower().strip()
                full_name = info.get("name") or info.get("given_name") or (email.split("@")[0] if email else "")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"تعذر التحقق من توكن Google: {e}",
            )
    elif payload.email:
        email = payload.email.lower().strip()
        full_name = payload.name.strip() if payload.name else email.split("@")[0]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="يجب إرسال رمز اعتماد Google (credential) أو بريد إلكتروني صالح.",
        )

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لم يتم العثور على بريد إلكتروني صالح في حساب Google.",
        )

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(UserModel).where(UserModel.email == email))
        user = result.scalar_one_or_none()

        if not user:
            random_pw_hash = hash_password(f"google_oauth_{uuid.uuid4().hex}")
            user = UserModel(
                email=email,
                phone=None,
                full_name=full_name or "مستخدم Google",
                password_hash=random_pw_hash,
                role=payload.role.value,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

        access_token = create_access_token(user.id, user.email, user.role)
        refresh_token = create_refresh_token(user.id)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MIN * 60,
        )


@auth_router.get("/me", response_model=UserProfileResponse)
async def get_me(current_user: UserModel = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        phone=current_user.phone,
        role=current_user.role,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
    )


@session_router.get("/me")
async def get_my_triage_sessions(current_user: UserModel = Depends(get_current_user)):
    """Returns the authenticated user's past triage sessions."""
    sessions = await list_user_sessions(current_user.id)
    return {"sessions": sessions, "count": len(sessions)}
