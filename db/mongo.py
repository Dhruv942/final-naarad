import motor.motor_asyncio
from urllib.parse import quote_plus

# MongoDB Configuration - Use local for development (no SSL issues)
# For production, use the commented Atlas connection
MONGO_DETAILS = "mongodb://localhost:27017"

# Production MongoDB Atlas (uncomment when SSL fixed)
# username = quote_plus("dhruvvpatel1010_db_user")
# password = quote_plus("Naarad@ss37")
# MONGO_DETAILS = f"mongodb+srv://{username}:{password}@production-naarad.x7gpsod.mongodb.net/?retryWrites=true&w=majority&appName=production-naarad&ssl=true&ssl_cert_reqs=CERT_NONE"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client.stagin_local

# Collections
users_collection = db.get_collection("users")
notifications_collection = db.get_collection("notifications")
alerts_collection = db.get_collection("alerts")
news_collection = db.get_collection("news")
feed_collection = db.get_collection("feeds")
personalized_collection = db.get_collection("personalized_news")

# RAG-specific collections
document_vectors_collection = db.get_collection("document_vectors")
user_profiles_collection = db.get_collection("user_profiles")
user_interactions_collection = db.get_collection("user_interactions")
user_feedback_collection = db.get_collection("user_feedback")