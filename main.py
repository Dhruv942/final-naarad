from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import asyncio
from contextlib import asynccontextmanager
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noisy third-party request logs (HTTP Request: GET ...)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # RAG system startup - no more old ML models
    logger.info("🚀 Starting Naarad with RAG Intelligence System")

    # Start the news alert scheduler in background (30m ON, 30m OFF cycle)
    from schedulers.news_pipeline_scheduler import start_news_pipeline_alternating
    scheduler_task = asyncio.create_task(start_news_pipeline_alternating(run_minutes=30, sleep_minutes=30))
    logger.info("📅 News Alert Scheduler started in background (30m ON, 30m OFF)")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Naarad RAG system")
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    logger.info("📅 News Alert Scheduler stopped")

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
    allow_origins=["*"],  # future me specific domains daalna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes import
from routes import user_routes
from routes import notification_prefernces
from routes import alerts
from controllers import gatekeeper
from controllers import rag_news_controller  # RAG controller
from controllers import feedback_controller  # AI/ML Feedback
from controllers import news_pipeline_controller  # 3-Stage LLM Pipeline

# Routers
app.include_router(user_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(notification_prefernces.router, prefix="/preferences", tags=["Notification Preferences"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(rag_news_controller.router, prefix="/rag", tags=["RAG News Intelligence"])  # RAG system
app.include_router(feedback_controller.router, prefix="/feedback", tags=["AI/ML Feedback & Learning"])  # Learning system
app.include_router(gatekeeper.router, prefix="/gatekeeper", tags=["Gatekeeper"])
app.include_router(news_pipeline_controller.router, prefix="/pipeline", tags=["News Pipeline"])  # 3-Stage Pipeline

# Mount static files (for serving images)
# Ensure directory exists to avoid RuntimeError when missing
static_dir = "static"
if not os.path.isdir(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Healthcheck
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "News App API is running"}

# Root
@app.get("/")
async def root():
    return {"message": "Welcome to News App API narrad"}


