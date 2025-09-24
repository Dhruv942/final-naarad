from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scheduler instance
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import here to avoid circular imports
    from controllers.news import process_scheduled_alerts

    # Add job to run every 2 minutes
    scheduler.add_job(
        process_scheduled_alerts,
        'interval',
        minutes=1,
        id='alert_processing',
        replace_existing=True
    )

    # Start scheduler
    scheduler.start()
    logger.info("Started automatic alert processing scheduler (every 2 minutes)")

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("Stopped alert processing scheduler")

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
from controllers import news
from controllers import gatekeeper
# Routers
app.include_router(user_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(notification_prefernces.router, prefix="/preferences", tags=["Notification Preferences"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(news.router, prefix="/news", tags=["News Cron"])  # 👈 cron wala router
app.include_router(gatekeeper.router, prefix="/gatekeeper", tags=["Gatekeeper"])


# Healthcheck
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "News App API is running"}

# Root
@app.get("/")
async def root():
    return {"message": "Welcome to News App API"}
