import motor.motor_asyncio
from pymongo.server_api import ServerApi

# MongoDB Configuration - Atlas connection
uri = "mongodb+srv://dhruvvpatel1010_db_user:yODfw0sTRAX6nFzw@naarad.xyujaxp.mongodb.net/?retryWrites=true&w=majority&appName=naarad"

# Local MongoDB for development (uncomment if needed)
# uri = "mongodb://localhost:27017"

client = motor.motor_asyncio.AsyncIOMotorClient(uri, server_api=ServerApi('1'))
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