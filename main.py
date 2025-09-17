from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Naarad App API", description="Personalized news aggregation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes import user_routes 
from routes import notification_prefernces
from routes import alerts
from controllers import news
app.include_router(user_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(notification_prefernces.router, prefix="/preferences", tags=["Notification Preferences"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(news.router, prefix="/news", tags=["News"])
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "News App API is running"}

@app.get("/")
async def root():
    return {"message": "Welcome to News App API"}