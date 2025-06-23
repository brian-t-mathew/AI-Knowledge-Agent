from dotenv import load_dotenv
import motor.motor_asyncio
import os
from datetime import datetime, timezone
import logging as logger

load_dotenv()

client = None
db = None
user_collection = None
chat_history_collection = None

async def init_db():
    global client, db, user_collection, chat_history_collection
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            os.getenv("MONGODB_URI"),
            maxPoolSize=10,
            minPoolSize=2,
            serverSelectionTimeoutMS=5000
        )
        
        # Verify connection
        await client.admin.command('ping')
        
        db = client["chatbot_db"]
        user_collection = db["users"]
        chat_history_collection = db["chat_history"]
        
        # Create indexes with background=True
        await chat_history_collection.create_index("session_id", background=True)
        await chat_history_collection.create_index("user_id", background=True)
        await user_collection.create_index("email", unique=True, background=True)
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise RuntimeError("Database connection failed")

async def close_db():
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")

def get_user_collection():
    if user_collection is None:
        raise RuntimeError("User collection not initialized")
    return user_collection


def get_chat_history_collection():
    if chat_history_collection is None:
        raise RuntimeError("Chat history collection not initialized")
    return chat_history_collection


async def get_db():
    if not db:
        raise RuntimeError("Database not initialized")
    return db

async def save_chat(session_id, role, content, user_id="guest"):
    try:
        collection = get_chat_history_collection()
        await collection.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception as e:
        logger.error(f"Failed to save chat: {e}")
        raise

async def get_chat_history(session_id):
    try:
        collection = get_chat_history_collection()
        cursor = collection.find({"session_id": session_id}).sort("timestamp", 1)
        records = await cursor.to_list(length=1000)
        return [{"role": record["role"], "content": record["content"]} for record in records]
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return []

async def get_user_sessions(user_id):
    try:
        collection = get_chat_history_collection()
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$session_id",
                "last_updated": {"$max": "$timestamp"}
            }},
            {"$sort": {"last_updated": -1}},
            {"$limit": 50}
        ]
        results = await collection.aggregate(pipeline).to_list(length=50)
        sessions = []
        for r in results:
            ts = r["last_updated"]
            # If ts is naive, make it UTC-aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            sessions.append({
                "session_id": r["_id"],
                "timestamp": ts.isoformat()
            })
        return sessions
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}")
        return []
    
async def create_user(email, hashed_pw):
    try:
        await get_user_collection().insert_one({
            "email": email,
            "password": hashed_pw
        })
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise



async def get_user_by_email(email):
    try:
        return await get_user_collection().find_one({"email": email})
    except Exception as e:
        logger.error(f"Failed to get user by email: {e}")
        return None