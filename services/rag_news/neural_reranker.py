"""
Neural Reranker Module

This module provides functionality to re-rank search results using neural models
for improved relevance based on the query and document content.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class NeuralReranker:
    """Handles neural re-ranking of search results"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the neural reranker with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained cross-encoder model
        """
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model"""
        try:
            logger.info(f"Loading neural reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Neural reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading neural reranker model: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of document dictionaries with at least 'content' key
            top_k: Number of top results to return (None for all)
            score_threshold: Minimum score threshold for inclusion in results
            
        Returns:
            List of re-ranked documents with 'score' field added
        """
        if not self.model:
            logger.error("Model not loaded for neural reranking")
            return documents
            
        if not documents:
            return []
        
        try:
            # Prepare input pairs for the cross-encoder
            pairs = [[query, doc.get("content", "")] for doc in documents]
            
            # Get scores from the model
            scores = self.model.predict(pairs)
            
            # Add scores to documents and filter by threshold
            for doc, score in zip(documents, scores):
                doc["score"] = float(score)
            
            # Sort by score (descending)
            sorted_docs = sorted(
                [doc for doc in documents if doc["score"] >= score_threshold],
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Return top-k results if specified
            return sorted_docs[:top_k] if top_k is not None else sorted_docs
            
        except Exception as e:
            logger.error(f"Error during neural reranking: {e}")
            return documents  # Return original order in case of error
    
    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Re-rank multiple sets of documents for multiple queries in batch.
        
        Args:
            queries: List of search queries
            documents_list: List of document lists, one per query
            top_k: Number of top results to return per query
            score_threshold: Minimum score threshold for inclusion in results
            
        Returns:
            List of re-ranked document lists, one per query
        """
        if not self.model:
            logger.error("Model not loaded for neural reranking")
            return documents_list
            
        results = []
        for query, docs in zip(queries, documents_list):
            reranked = await self.rerank(query, docs, top_k, score_threshold)
            results.append(reranked)
            
        return results


# Singleton instance
_neural_reranker = None

def get_neural_reranker(model_name: str = None) -> NeuralReranker:
    """Get or create a NeuralReranker instance"""
    global _neural_reranker
    if _neural_reranker is None:
        _neural_reranker = NeuralReranker(model_name) if model_name else NeuralReranker()
    return _neural_reranker
