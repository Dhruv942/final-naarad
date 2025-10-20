"""
Clear sent notifications for testing
Usage:
    python clear_sent_notifications.py <user_id>
    python clear_sent_notifications.py all  (to clear all users)
"""
import asyncio
import sys
from db.mongo import db

async def clear_sent(user_id=None):
    coll = db.get_collection("sent_notifications")
    
    # Determine filter
    if user_id and user_id.lower() != "all":
        filter_query = {"user_id": user_id}
        print(f"🎯 Target: User {user_id}")
    else:
        filter_query = {}
        print(f"🎯 Target: ALL USERS")
    
    # Count before
    before = await coll.count_documents(filter_query)
    print(f"📊 Sent notifications before: {before}")
    
    if before == 0:
        print("ℹ️  No notifications to clear!")
        return
    
    # Delete
    result = await coll.delete_many(filter_query)
    print(f"✅ Deleted: {result.deleted_count} notifications")
    
    # Count after
    after = await coll.count_documents(filter_query)
    print(f"📊 Sent notifications after: {after}")
    
    if after == 0:
        print("\n✅ SUCCESS! All notifications cleared.")
        print("Now call the API again to get fresh articles!")
    else:
        print("\n⚠️  Some notifications remain")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python clear_sent_notifications.py <user_id>")
        print("  python clear_sent_notifications.py all")
        print("\nExample:")
        print("  python clear_sent_notifications.py 14a9d3ec-a283-47a8-ab3a-9f974a4c3ea5")
        sys.exit(1)
    
    user_id = sys.argv[1]
    asyncio.run(clear_sent(user_id))
