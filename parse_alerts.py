import asyncio
from db.mongo import client, alerts_collection
from bson import ObjectId

async def parse_alerts():
    # Create the alertspars collection if it doesn't exist
    db = client.stagin_local
    alertspars_collection = db.alertspars
    
    # Clear existing parsed alerts if needed
    # await alertspars_collection.delete_many({})
    
    # Process each alert in the alerts collection
    async for alert in alerts_collection.find():
        # Extract fields from the original alert
        alert_id = str(alert.get("_id"))
        preferences = alert.get("preferences", [{}])[0]  # Get first preference object
        
        # Create the parsed alert structure
        parsed_alert = {
            "alert_id": alert_id,
            "category": preferences.get("category", ""),
            "sub_categories": preferences.get("sub_categories", []),
            "followup_questions": preferences.get("followup_questions", []),
            "custom_question": preferences.get("custom_question", ""),
            "event_conditions": preferences.get("event_conditions", []),
            "contextual_query": preferences.get("contextual_query", ""),
            "original_alert": alert  # Keep the original alert data for reference
        }
        
        # Insert the parsed alert into the new collection
        await alertspars_collection.update_one(
            {"alert_id": str(alert["_id"])},
            {"$set": parsed_alert},
            upsert=True
        )
        print(f"Processed alert: {alert['_id']}")

async def main():
    try:
        await parse_alerts()
        print("Alert parsing completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
