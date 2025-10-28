"""
Enhanced RAG (Retrieval-Augmented Generation) System
- Improved performance with batch processing
- Better error handling and logging
- Cleaner code structure
- Support for multiple embedding models
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, TypeVar, Generic, Type
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from pymongo import UpdateOne, IndexModel, ASCENDING, DESCENDING
from sentence_transformers import SentenceTransformer
import httpx
from pydantic import BaseModel, Field, validator
from pymongo.database import Database
from pymongo.collection import Collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class DocumentType(str, Enum):
    ARTICLE = "article"
    NEWS = "news"
    BLOG = "blog"
    TWEET = "tweet"
    OTHER = "other"

class Document(BaseModel):
    """Enhanced document model with Pydantic validation"""
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., min_length=10, description="Main content of the document")
    title: str = Field(default="", description="Document title")
    url: str = Field(..., description="Source URL")
    category: str = Field(default="general", description="Document category")
    source: str = Field(default="", description="Source name")
    doc_type: DocumentType = Field(default=DocumentType.ARTICLE, description="Type of document")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation/update timestamp")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score (0-1)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
            DocumentType: lambda v: v.value
        }

class UserProfile(BaseModel):
    """Enhanced user profile with Pydantic validation"""
    user_id: str = Field(..., description="Unique user identifier")
    interests: List[str] = Field(default_factory=list, description="List of user interests")
    reading_history: List[str] = Field(default_factory=list, description="List of read document IDs")
    feedback_scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Document ID to feedback score mapping"
    )
    preferred_categories: List[str] = Field(
        default_factory=list, 
        description="User's preferred content categories"
    )
    preferred_sources: List[str] = Field(
        default_factory=list, 
        description="User's preferred news sources"
    )
    interaction_patterns: Dict[str, Any] = Field(
        default_factory=dict,
        description="User interaction patterns and preferences"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last profile update timestamp"
    )
    
    def update_from_interaction(self, doc_id: str, interaction_type: str, **kwargs):
        """Update profile based on user interaction"""
        self.last_updated = datetime.utcnow()
        
        # Update reading history (FIFO, max 100 items)
        if doc_id in self.reading_history:
            self.reading_history.remove(doc_id)
        self.reading_history.insert(0, doc_id)
        self.reading_history = self.reading_history[:100]
        
        # Update interaction patterns
        if interaction_type not in self.interaction_patterns:
            self.interaction_patterns[interaction_type] = {"count": 0, "last_interaction": None}
        
        self.interaction_patterns[interaction_type]["count"] += 1
        self.interaction_patterns[interaction_type]["last_interaction"] = datetime.utcnow()
        
        # Update feedback scores if provided
        if "feedback_score" in kwargs:
            self.feedback_scores[doc_id] = kwargs["feedback_score"]

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        pass

class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformer based embedding model"""
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously"""
        # Run in thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.model.encode(texts, show_progress_bar=False).tolist()
        )
    
    @property
    def dimension(self) -> int:
        return self._dimension

class RAGVectorStore:
    """
    Enhanced vector store with improved performance and error handling
    - Batch processing for bulk operations
    - Efficient similarity search with filters
    - Automatic index management
    """
    
    def __init__(self, collection: Collection, embedding_model: EmbeddingModel):
        self.collection = collection
        self.embedding_model = embedding_model
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary indexes for efficient querying"""
        try:
            # Create text index for search
            self.collection.create_index(
                [("content", "text"), ("title", "text")], 
                name="text_search_idx",
                default_language="english"
            )
            
            # Create other indexes
            self.collection.create_indexes([
                IndexModel([("timestamp", DESCENDING)], name="timestamp_idx"),
                IndexModel([("category", ASCENDING)], name="category_idx"),
                IndexModel([("source", ASCENDING)], name="source_idx")
            ])
            logger.info("Successfully created database indexes")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> int:
        """Add multiple documents with embeddings to the store"""
        if not documents:
            return 0
            
        # Generate embeddings in batches
        batch_size = 32
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.content for doc in batch]
            
            try:
                # Generate embeddings
                embeddings = await self.embedding_model.embed(texts)
                
                # Prepare documents for insertion
                docs_to_insert = []
                for doc, embedding in zip(batch, embeddings):
                    doc_dict = doc.dict()
                    doc_dict["embedding"] = embedding
                    doc_dict["last_updated"] = datetime.utcnow()
                    docs_to_insert.append(doc_dict)
                
                # Bulk insert
                result = await self.collection.insert_many(docs_to_insert)
                total_added += len(result.inserted_ids)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
                
        return total_added
    
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict] = None,
        min_score: float = 0.5
    ) -> List[Document]:
        """
        Perform similarity search with filters and score thresholding
        
        Args:
            query: The search query
            k: Number of results to return
            filters: MongoDB query filters
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching documents with relevance scores
        """
        # Generate query embedding
        query_embedding = (await self.embedding_model.embed([query]))[0]
        
        # Build the aggregation pipeline
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": k * 2  # Fetch more to account for filtering
                    },
                    "returnStoredSource": True
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "searchScore"}
                }
            },
            {"$match": {"score": {"$gte": min_score}}},
            {"$limit": k}
        ]
        
        # Add additional filters if provided
        if filters:
            pipeline.insert(1, {"$match": filters})
        
        try:
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=k)
            
            # Convert to Document objects
            return [
                Document(
                    **{k: v for k, v in doc.items() 
                      if k != '_id' and not k.startswith('_')}
                )
                for doc in results
            ]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

class RAGSystem:
    """
    Enhanced RAG system with improved personalization and performance
    """
    
    def __init__(
        self, 
        db: Database,
        embedding_model: Optional[EmbeddingModel] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.db = db
        self.gemini_api_key = gemini_api_key
        
        # Initialize embedding model
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        
        # Initialize collections
        self.vector_store = RAGVectorStore(
            self.db.get_collection("document_vectors"),
            self.embedding_model
        )
        
        self.user_profiles = self.db.get_collection("user_profiles")
        self.feedback = self.db.get_collection("user_feedback")
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
    
    async def process_documents(self, documents: List[Dict]) -> int:
        """Process and store multiple documents"""
        # Convert to Document objects
        docs = [
            Document(
                id=doc.get("id") or str(hashlib.md5(doc["content"].encode()).hexdigest()),
                **{k: v for k, v in doc.items() if k != "id"}
            )
            for doc in documents
            if doc.get("content")
        ]
        
        return await self.vector_store.add_documents(docs)
    
    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for documents with optional personalization
        
        Args:
            query: The search query
            user_id: Optional user ID for personalization
            k: Number of results to return
            filters: Additional filters to apply
            
        Returns:
            List of relevant documents
        """
        # Get user profile if user_id is provided
        user_profile = None
        if user_id:
            user_profile = await self._get_user_profile(user_id)
            
            # Enhance query with user preferences
            if user_profile:
                query = self._enhance_query_with_profile(query, user_profile)
                
                # Add user's preferred categories/sources to filters
                filters = filters or {}
                if user_profile.preferred_categories:
                    filters["category"] = {"$in": user_profile.preferred_categories}
                if user_profile.preferred_sources:
                    filters["source"] = {"$in": user_profile.preferred_sources}
        
        # Perform the search
        results = await self.vector_store.similarity_search(
            query=query,
            k=k,
            filters=filters
        )
        
        # Record interaction if user is logged in
        if user_id and results:
            await self._record_search_interaction(user_id, query, results)
        
        return results
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile, creating a new one if it doesn't exist"""
        profile = await self.user_profiles.find_one({"user_id": user_id})
        if profile:
            return UserProfile(**profile)
        return None
    
    def _enhance_query(self, query: str, user_profile: UserProfile) -> str:
        """Enhance query with user profile information"""
        # Add user interests to query
        if user_profile.interests:
            query += " " + " ".join(user_profile.interests[:3])
            
        # Add recent feedback terms
        recent_feedback = [
            doc_id for doc_id, score in sorted(
                user_profile.feedback_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        ]
        
        if recent_feedback:
            # In a real system, you might want to fetch document terms here
            query += " " + " ".join(recent_feedback)
            
        return query.strip()
    
    async def _record_search_interaction(
        self,
        user_id: str,
        query: str,
        results: List[Document]
    ):
        """Record user search interaction"""
        interaction = {
            "user_id": user_id,
            "query": query,
            "results": [r.id for r in results],
            "timestamp": datetime.utcnow(),
            "result_count": len(results)
        }
        
        try:
            await self.db.user_searches.insert_one(interaction)
        except Exception as e:
            logger.error(f"Error recording search interaction: {str(e)}")
    
    async def record_feedback(
        self,
        user_id: str,
        doc_id: str,
        score: float,
        feedback_type: str = "relevance"
    ) -> bool:
        """Record user feedback on a document"""
        feedback = {
            "user_id": user_id,
            "doc_id": doc_id,
            "score": max(0.0, min(1.0, score)),  # Clamp to [0, 1]
            "type": feedback_type,
            "timestamp": datetime.utcnow()
        }
        
        try:
            # Store feedback
            await self.feedback.insert_one(feedback)
            
            # Update user profile
            await self.user_profiles.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        f"feedback_scores.{doc_id}": score,
                        "last_updated": datetime.utcnow()
                    },
                    "$addToSet": {
                        "reading_history": {"$each": [doc_id], "$slice": -100}
                    }
                },
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return False

# Example usage
async def example_usage():
    from motor.motor_asyncio import AsyncIOMotorClient
    
    # Initialize MongoDB client
    mongo_uri = "mongodb://localhost:27017"
    client = AsyncIOMotorClient(mongo_uri)
    db = client.rag_demo
    
    # Initialize RAG system
    rag = RAGSystem(db)
    
    try:
        # Example documents
        documents = [
            {
                "title": "Introduction to RAG",
                "content": "Retrieval-Augmented Generation combines retrieval and generation...",
                "url": "https://example.com/rag-intro",
                "category": "ai",
                "source": "example"
            },
            # Add more documents...
        ]
        
        # Process documents
        count = await rag.process_documents(documents)
        print(f"Processed {count} documents")
        
        # Search
        results = await rag.search("what is RAG?", k=5)
        for doc in results:
            print(f"- {doc.title} (score: {doc.relevance_score:.3f})")
            
    finally:
        # Clean up
        await rag.close()
        client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
