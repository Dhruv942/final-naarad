"""
RAG (Retrieval-Augmented Generation) System for Personalized News Intelligence
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import UpdateOne
import httpx

# Configuration
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document structure for RAG system"""
    id: str
    content: str
    title: str
    url: str
    category: str
    source: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    relevance_score: Optional[float] = None

@dataclass
class UserProfile:
    """User profile for personalization"""
    user_id: str
    interests: List[str]
    reading_history: List[str]
    feedback_scores: Dict[str, float]
    preferred_categories: List[str]
    preferred_sources: List[str]
    interaction_patterns: Dict[str, Any]
    last_updated: datetime

class RAGVectorStore:
    """Vector database for document storage and retrieval"""

    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.dimension = 384  # MiniLM-L6-v2 embedding dimension

    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector store with embeddings"""
        try:
            operations = []
            for doc in documents:
                # Generate embedding if not present
                if doc.embedding is None:
                    doc.embedding = self.embedding_model.encode(
                        f"{doc.title} {doc.content}"
                    ).tolist()

                doc_dict = asdict(doc)
                doc_dict['_id'] = doc.id

                operations.append(
                    UpdateOne(
                        {'_id': doc.id},
                        {'$set': doc_dict},
                        upsert=True
                    )
                )

            if operations:
                await self.collection.bulk_write(operations)
                logger.info(f"Added {len(operations)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_criteria: Optional[Dict] = None
    ) -> List[Document]:
        """Perform similarity search using cosine similarity"""
        try:
            # Build aggregation pipeline
            pipeline = []

            # Add filter if provided
            if filter_criteria:
                pipeline.append({"$match": filter_criteria})

            # Add similarity computation
            pipeline.extend([
                {
                    "$addFields": {
                        "similarity": {
                            "$let": {
                                "vars": {
                                    "dot_product": {
                                        "$sum": {
                                            "$map": {
                                                "input": {"$range": [0, len(query_embedding)]},
                                                "as": "i",
                                                "in": {
                                                    "$multiply": [
                                                        {"$arrayElemAt": ["$embedding", "$$i"]},
                                                        query_embedding["$$i"] if isinstance(query_embedding, dict) else query_embedding[0]  # Handle both formats
                                                    ]
                                                }
                                            }
                                        }
                                    }
                                },
                                "in": "$$dot_product"
                            }
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": k}
            ])

            # Simplified approach using client-side computation for reliability
            cursor = self.collection.find(filter_criteria or {})
            candidates = []

            async for doc in cursor:
                if 'embedding' in doc and doc['embedding']:
                    # Compute cosine similarity
                    doc_embedding = np.array(doc['embedding'])
                    query_vec = np.array(query_embedding)

                    # Normalize vectors
                    doc_norm = np.linalg.norm(doc_embedding)
                    query_norm = np.linalg.norm(query_vec)

                    if doc_norm > 0 and query_norm > 0:
                        similarity = np.dot(doc_embedding, query_vec) / (doc_norm * query_norm)
                        doc['similarity'] = float(similarity)
                        candidates.append(doc)

            # Sort by similarity and take top k
            candidates.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            top_candidates = candidates[:k]

            # Convert to Document objects
            results = []
            for doc in top_candidates:
                document = Document(
                    id=doc['_id'],
                    content=doc.get('content', ''),
                    title=doc.get('title', ''),
                    url=doc.get('url', ''),
                    category=doc.get('category', ''),
                    source=doc.get('source', ''),
                    embedding=doc.get('embedding'),
                    metadata=doc.get('metadata', {}),
                    timestamp=doc.get('timestamp'),
                    relevance_score=doc.get('similarity', 0)
                )
                results.append(document)

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

class PersonalizationEngine:
    """Handles user profiling and personalization"""

    def __init__(self, users_collection, feedback_collection, embedding_model):
        self.users_collection = users_collection
        self.feedback_collection = feedback_collection
        self.embedding_model = embedding_model

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile"""
        try:
            user_doc = await self.users_collection.find_one({"_id": user_id})
            if not user_doc:
                return None

            return UserProfile(
                user_id=user_id,
                interests=user_doc.get('interests', []),
                reading_history=user_doc.get('reading_history', []),
                feedback_scores=user_doc.get('feedback_scores', {}),
                preferred_categories=user_doc.get('preferred_categories', []),
                preferred_sources=user_doc.get('preferred_sources', []),
                interaction_patterns=user_doc.get('interaction_patterns', {}),
                last_updated=user_doc.get('last_updated', datetime.utcnow())
            )
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None

    async def update_user_profile(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> bool:
        """Update user profile based on interactions"""
        try:
            update_data = {
                'last_updated': datetime.utcnow()
            }

            # Update reading history
            if 'read_articles' in interaction_data:
                update_data['$push'] = {
                    'reading_history': {
                        '$each': interaction_data['read_articles'],
                        '$slice': -100  # Keep last 100 articles
                    }
                }

            # Update preferences based on feedback
            if 'feedback' in interaction_data:
                for article_id, score in interaction_data['feedback'].items():
                    update_data[f'feedback_scores.{article_id}'] = score

            await self.users_collection.update_one(
                {'_id': user_id},
                {'$set': update_data},
                upsert=True
            )
            return True

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    def generate_personalized_query(
        self,
        user_profile: UserProfile,
        base_query: str
    ) -> str:
        """Generate personalized query based on user profile"""
        personalized_parts = [base_query]

        # Add user interests
        if user_profile.interests:
            interests_text = " ".join(user_profile.interests)
            personalized_parts.append(interests_text)

        # Add preferred categories context
        if user_profile.preferred_categories:
            categories_text = " ".join(user_profile.preferred_categories)
            personalized_parts.append(categories_text)

        return " ".join(personalized_parts)

class RAGSystem:
    """Main RAG system coordinating all components"""

    def __init__(self, mongo_db, gemini_api_key: str):
        self.db = mongo_db
        self.gemini_api_key = gemini_api_key
        self.gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize components
        self.vector_store = RAGVectorStore(
            self.db.get_collection("document_vectors"),
            self.embedding_model
        )

        self.personalization_engine = PersonalizationEngine(
            self.db.get_collection("user_profiles"),
            self.db.get_collection("user_feedback"),
            self.embedding_model
        )

        # Collections
        self.news_collection = self.db.get_collection("processed_news")
        self.cache_collection = self.db.get_collection("rag_cache")

    async def process_news_articles(self, articles: List[Dict]) -> bool:
        """Process and store news articles in vector database"""
        try:
            documents = []
            for article in articles:
                # Create document ID
                doc_id = hashlib.md5(
                    f"{article.get('url', '')}{article.get('title', '')}".encode()
                ).hexdigest()

                # Extract content and metadata
                content = f"{article.get('summary', '')} {article.get('description', '')}"

                document = Document(
                    id=doc_id,
                    content=content,
                    title=article.get('title', ''),
                    url=article.get('url', ''),
                    category=article.get('category', 'general'),
                    source=article.get('source', ''),
                    metadata={
                        'published_date': article.get('published_date'),
                        'author': article.get('author'),
                        'tags': article.get('tags', [])
                    },
                    timestamp=datetime.utcnow()
                )
                documents.append(document)

            # Store in vector database
            success = await self.vector_store.add_documents(documents)

            # Also store in news collection for reference
            if success and documents:
                news_docs = [asdict(doc) for doc in documents]
                operations = [
                    UpdateOne(
                        {'_id': doc['id']},
                        {'$set': doc},
                        upsert=True
                    ) for doc in news_docs
                ]
                await self.news_collection.bulk_write(operations)

            return success

        except Exception as e:
            logger.error(f"Error processing news articles: {e}")
            return False

    async def retrieve_personalized_news(
        self,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve personalized news using RAG"""
        try:
            # Get user profile
            user_profile = await self.personalization_engine.get_user_profile(user_id)

            # Generate personalized query
            if user_profile:
                personalized_query = self.personalization_engine.generate_personalized_query(
                    user_profile, query
                )
            else:
                personalized_query = query

            # Generate query embedding
            query_embedding = self.embedding_model.encode(personalized_query).tolist()

            # Build filter criteria based on user preferences
            filter_criteria = {}
            if user_profile:
                if user_profile.preferred_categories:
                    filter_criteria['category'] = {'$in': user_profile.preferred_categories}
                if user_profile.preferred_sources:
                    filter_criteria['source'] = {'$in': user_profile.preferred_sources}

            # Retrieve similar documents
            similar_docs = await self.vector_store.similarity_search(
                query_embedding,
                k=limit * 2,  # Get more to filter later
                filter_criteria=filter_criteria
            )

            # Re-rank based on user feedback if available
            if user_profile and user_profile.feedback_scores:
                similar_docs = self._rerank_with_feedback(similar_docs, user_profile)

            # Prepare response
            results = []
            for doc in similar_docs[:limit]:
                result = {
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content,
                    'url': doc.url,
                    'category': doc.category,
                    'source': doc.source,
                    'relevance_score': doc.relevance_score,
                    'metadata': doc.metadata,
                    'timestamp': doc.timestamp
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error retrieving personalized news: {e}")
            return []

    def _rerank_with_feedback(
        self,
        documents: List[Document],
        user_profile: UserProfile
    ) -> List[Document]:
        """Re-rank documents based on user feedback"""
        try:
            for doc in documents:
                # Base score from similarity
                base_score = doc.relevance_score or 0

                # Boost based on category preference
                category_boost = 0
                if doc.category in user_profile.preferred_categories:
                    category_boost = 0.1

                # Boost based on source preference
                source_boost = 0
                if doc.source in user_profile.preferred_sources:
                    source_boost = 0.1

                # Boost based on historical feedback
                feedback_boost = 0
                similar_articles = [
                    article_id for article_id in user_profile.feedback_scores
                    if any(keyword in doc.title.lower()
                          for keyword in article_id.split('_')[-3:])  # Simple similarity check
                ]
                if similar_articles:
                    avg_feedback = np.mean([
                        user_profile.feedback_scores[article_id]
                        for article_id in similar_articles
                    ])
                    feedback_boost = (avg_feedback - 0.5) * 0.2  # Scale feedback

                # Calculate final score
                doc.relevance_score = base_score + category_boost + source_boost + feedback_boost

            # Sort by updated relevance score
            documents.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            return documents

        except Exception as e:
            logger.error(f"Error re-ranking with feedback: {e}")
            return documents

    async def generate_intelligent_summary(
        self,
        documents: List[Document],
        user_query: str
    ) -> str:
        """Generate intelligent summary using retrieved documents"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for doc in documents[:5]:  # Use top 5 documents
                context_parts.append(f"Title: {doc.title}\nContent: {doc.content[:500]}...")

            context = "\n\n".join(context_parts)

            # Prepare prompt for Gemini
            prompt = f"""
            Based on the following news articles, provide a comprehensive and personalized summary for the user query: "{user_query}"

            News Articles:
            {context}

            Please provide:
            1. A concise summary of the key information
            2. Relevant insights and connections between the articles
            3. Personalized recommendations based on the content

            Summary:
            """

            # Call Gemini API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.gemini_endpoint,
                    headers={
                        'Content-Type': 'application/json',
                    },
                    params={'key': self.gemini_api_key},
                    json={
                        'contents': [{
                            'parts': [{'text': prompt}]
                        }]
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    summary = result['candidates'][0]['content']['parts'][0]['text']
                    return summary
                else:
                    logger.error(f"Gemini API error: {response.status_code}")
                    return "Unable to generate summary at this time."

        except Exception as e:
            logger.error(f"Error generating intelligent summary: {e}")
            return "Error generating summary."

    async def record_user_interaction(
        self,
        user_id: str,
        article_id: str,
        interaction_type: str,
        feedback_score: Optional[float] = None
    ) -> bool:
        """Record user interaction for continuous learning"""
        try:
            interaction_data = {
                'user_id': user_id,
                'article_id': article_id,
                'interaction_type': interaction_type,  # 'view', 'like', 'share', 'bookmark'
                'feedback_score': feedback_score,
                'timestamp': datetime.utcnow()
            }

            # Store interaction
            await self.db.get_collection("user_interactions").insert_one(interaction_data)

            # Update user profile
            update_data = {}
            if interaction_type == 'view':
                update_data['read_articles'] = [article_id]
            elif feedback_score is not None:
                update_data['feedback'] = {article_id: feedback_score}

            if update_data:
                await self.personalization_engine.update_user_profile(user_id, update_data)

            return True

        except Exception as e:
            logger.error(f"Error recording user interaction: {e}")
            return False