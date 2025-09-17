import motor.motor_asyncio

MONGO_DETAILS = "mongodb://localhost:27017"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client.stagin_local
users_collection = db.get_collection("users")
notifications_collection = db.get_collection("notifications")
alerts_collection = db.get_collection("alerts")
news_collection = db.get_collection("news")
feed_collection = db.get_collection("feeds")
personalized_collection = db.get_collection("personalized_news")