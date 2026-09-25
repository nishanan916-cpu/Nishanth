"""
app.py
------
PocketSmart AI - main FastAPI application.

Handles:
- Static files + Jinja2 templates
- Simple in-memory user registration / login (JWT via cookie)
- Session tracking
- Three planner endpoints (home / party / jewelry) that call gemini_utils.py
- Recommendation history endpoints
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile, Form,
    Request, status, Cookie
)
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

from gemini_utils import (
    HomeBudgetInput, PartyBudgetInput, JewelryBudgetInput,
    get_home_recommendations, get_party_recommendations, get_jewelry_recommendations,
    save_upload_file, save_to_history, user_recommendations,
)

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

app = FastAPI(title="PocketSmart AI: Budget Planner")

SECRET_KEY = os.getenv("SECRET_KEY", "insecure-dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/uploads", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------------------------------------------------------------
# In-memory "database" (swap for a real DB in production)
# --------------------------------------------------------------------------

class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str


class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: Dict[str, Any] = {}


users_db: Dict[str, dict] = {}
active_sessions: Dict[str, UserSession] = {}
blacklisted_tokens: set = set()


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(db: dict, username: str, password: str):
    user = db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return UserInDB(**user)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_token(request: Request) -> Optional[str]:
    """Read the JWT either from the Authorization header or the access_token cookie."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    return request.cookies.get("access_token")


async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token or token in blacklisted_tokens:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_db.get(username)
    if user is None:
        raise credentials_exception

    if username in active_sessions:
        active_sessions[username].last_activity = datetime.utcnow()

    return UserInDB(**user)


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    return current_user


async def get_optional_user(request: Request) -> Optional[UserInDB]:
    """Like get_current_user, but returns None instead of raising (for public pages)."""
    token = await get_token(request)
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    user = users_db.get(username)
    return UserInDB(**user) if user else None


# --------------------------------------------------------------------------
# Public pages
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: Optional[UserInDB] = Depends(get_optional_user)):
    """Landing page."""
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: Optional[UserInDB] = Depends(get_optional_user)):
    """Serve the login page."""
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request, "user": None})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: Optional[UserInDB] = Depends(get_optional_user)):
    """Serve the registration page."""
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {"request": request, "user": None})


# --------------------------------------------------------------------------
# Auth API endpoints
# --------------------------------------------------------------------------

@app.post("/api/register")
async def api_register(payload: RegisterUser):
    """Create a new user account."""
    if payload.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")

    users_db[payload.username] = {
        "username": payload.username,
        "email": payload.email,
        "full_name": payload.full_name,
        "hashed_password": get_password_hash(payload.password),
    }
    return {"message": "Account created successfully. Please sign in."}


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get an access token (also sets it as a cookie)."""
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    existing_user_data = {}
    if user.username in active_sessions:
        old_token = active_sessions[user.username].token
        blacklisted_tokens.add(old_token)
        existing_user_data = active_sessions[user.username].user_data

    active_sessions[user.username] = UserSession(
        username=user.username,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        token=access_token,
        user_data=existing_user_data,
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@app.post("/logout")
async def logout(request: Request):
    """Logout user by blacklisting their token and clearing the session."""
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass

    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response


# --------------------------------------------------------------------------
# Protected pages
# --------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """User dashboard showing recent activity."""
    recent = sorted(
        user_recommendations.get(current_user.username, []),
        key=lambda r: r.timestamp,
        reverse=True,
    )[:5]
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": current_user, "recent": recent},
    )


@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Home budget planner page."""
    return templates.TemplateResponse("home_planner.html", {"request": request, "user": current_user})


@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Party budget planner page."""
    return templates.TemplateResponse("party_planner.html", {"request": request, "user": current_user})


@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Jewelry budget planner page."""
    return templates.TemplateResponse("jewelry_planner.html", {"request": request, "user": current_user})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """History page to view past recommendations."""
    return templates.TemplateResponse("history.html", {"request": request, "user": current_user})


# --------------------------------------------------------------------------
# Planner API endpoints
# --------------------------------------------------------------------------

@app.post("/home-budget")
async def plan_home_budget(
    budget_input: HomeBudgetInput,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate home budget recommendations."""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
        }

    try:
        result = get_home_recommendations(budget_input)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    save_to_history(
        username=current_user.username,
        recommendation_type="home",
        input_data=budget_input.model_dump(),
        result=result,
    )
    return result


@app.post("/party-budget")
async def plan_party_budget(
    budget_input: PartyBudgetInput,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate party budget recommendations."""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
        }

    try:
        result = get_party_recommendations(budget_input)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    save_to_history(
        username=current_user.username,
        recommendation_type="party",
        input_data=budget_input.model_dump(),
        result=result,
    )
    return result


@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate jewelry recommendations, optionally analyzing an uploaded outfit image."""
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences,
    )

    image_path = None
    if image and image.filename:
        image_path = save_upload_file(image)

    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_jewelry_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "occasion": budget_input.occasion,
            "has_image": image_path is not None,
        }

    try:
        result = get_jewelry_recommendations(budget_input, image_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    input_data = budget_input.model_dump()
    if image_path:
        input_data["image"] = image.filename

    save_to_history(
        username=current_user.username,
        recommendation_type="jewelry",
        input_data=input_data,
        result=result,
    )
    return result


# --------------------------------------------------------------------------
# History API endpoints
# --------------------------------------------------------------------------

@app.get("/recommendation-history")
async def get_recommendation_history(current_user: UserInDB = Depends(get_current_active_user)):
    """Get the user's recommendation history."""
    if current_user.username not in user_recommendations:
        return {"history": []}

    history = sorted(
        user_recommendations[current_user.username],
        key=lambda x: x.timestamp,
        reverse=True,
    )

    history_data = [
        {
            "id": item.id,
            "timestamp": item.timestamp,
            "type": item.recommendation_type,
            "input": item.input_summary,
            "summary": item.result_summary,
        }
        for item in history
    ]
    return {"history": history_data}


@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Get the full details of a specific recommendation."""
    if current_user.username not in user_recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")

    for item in user_recommendations[current_user.username]:
        if item.id == recommendation_id:
            return {
                "id": item.id,
                "timestamp": item.timestamp,
                "type": item.recommendation_type,
                "input": item.input_summary,
                "full_result": item.full_result,
            }

    raise HTTPException(status_code=404, detail="Recommendation not found")


# --------------------------------------------------------------------------
# Session endpoints
# --------------------------------------------------------------------------

@app.get("/session-info")
async def get_session_info(current_user: UserInDB = Depends(get_current_active_user)):
    """Get current user's session information."""
    if current_user.username in active_sessions:
        session = active_sessions[current_user.username]
        return {
            "username": session.username,
            "login_time": session.login_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "session_duration_minutes": (datetime.utcnow() - session.login_time).total_seconds() // 60,
            "user_data": session.user_data,
        }
    raise HTTPException(status_code=404, detail="No active session found")


@app.post("/session-data")
async def update_session_data(
    data: Dict[str, Any],
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Update the current user's session data."""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data.update(data)
        active_sessions[current_user.username].last_activity = datetime.utcnow()
        return {"message": "Session data updated", "data": active_sessions[current_user.username].user_data}
    raise HTTPException(status_code=404, detail="No active session found")


@app.get("/api/me")
async def read_current_user(current_user: UserInDB = Depends(get_current_active_user)):
    """Return basic info about the logged-in user (used by the frontend nav bar)."""
    return {"username": current_user.username, "email": current_user.email, "full_name": current_user.full_name}


# --------------------------------------------------------------------------
# Background cleanup task
# --------------------------------------------------------------------------

@app.on_event("startup")
async def setup_session_cleanup():
    """Background task to clean up expired sessions."""

    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            expired_usernames = [
                username for username, session in list(active_sessions.items())
                if (current_time - session.last_activity).total_seconds() > 1800  # 30 minutes
            ]
            for username in expired_usernames:
                active_sessions.pop(username, None)
            await asyncio.sleep(300)  # check every 5 minutes

    asyncio.create_task(cleanup_expired_sessions())


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart AI Budget Planner...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
