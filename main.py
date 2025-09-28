from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # RAG system startup - no more old ML models
    logger.info("🚀 Starting Naarad with RAG Intelligence System")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Naarad RAG system")

# FastAPI App
app = FastAPI(
    title="Naarad App API",
    description="Personalized news aggregation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 future me specific domains daalna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes import
from routes import user_routes
from routes import notification_prefernces
from routes import alerts
from controllers import gatekeeper
from controllers import rag_news_controller  # 👈 RAG controller

# Routers
app.include_router(user_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(notification_prefernces.router, prefix="/preferences", tags=["Notification Preferences"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(rag_news_controller.router, prefix="/rag", tags=["RAG News Intelligence"])  # 👈 RAG system
app.include_router(gatekeeper.router, prefix="/gatekeeper", tags=["Gatekeeper"])


# Healthcheck
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "News App API is running"}

# Root
@app.get("/")
async def root():
    return {"message": "Welcome to News App API"}
