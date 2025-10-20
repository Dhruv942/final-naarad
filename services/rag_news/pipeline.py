import logging
from typing import Dict, Any, List

from .article_filter import (
    build_contextual_query,
    filter_recent_articles,
)
from .config_provider import get_effective_config
from .retriever import retrieve_candidates
from .reranker import rerank_candidates
from .diversity import mmr_diversify
from .reranker_ce import get_ce_reranker
from .popularity import popularity_proxy, time_decay
from .judge import get_gemini_judge

# New AI/ML components
from .nlp_intelligence import enrich_articles_with_intelligence, get_nlp_intelligence
from .personalization_engine import get_personalization_engine
from .neural_reranker import get_neural_reranker

logger = logging.getLogger(__name__)


async def run_pipeline(alert: Dict[str, Any], articles: List[dict], embedding_model) -> List[dict]:
    """
    Advanced AI/ML Pipeline v3 with:
    - NLP Intelligence (sentiment, entities, quality analysis)
    - User personalization and behavior learning
    - Neural reranking with feedback
    - Enhanced Gemini judge with chain-of-thought
    """
    try:
        category = (alert or {}).get("main_category", "").lower()
        keywords = (alert or {}).get("sub_categories", []) or []
        followups = (alert or {}).get("followup_questions", []) or []
        custom_q = (alert or {}).get("custom_question", "") or ""
        user_id = (alert or {}).get("user_id", "")

        logger.info(f"🚀 Starting intelligent pipeline for user {user_id}, category: {category}")

        # 1) Build contextual query
        contextual_query = await build_contextual_query(alert, keywords, followups, custom_q, category)

        # 2) Time filter with per-alert recency window (default 24h)
        recent_hours = int((alert or {}).get("recent_hours", 24))
        recent = filter_recent_articles(articles, max_hours=recent_hours)
        
        # 2b) NLP Intelligence: Enrich articles with sentiment, entities, quality scores
        logger.info("🧠 Enriching articles with NLP intelligence...")
        recent = enrich_articles_with_intelligence(recent)

        # 3) Dynamic config
        cfg = await get_effective_config(category, user_id, alert)
        retrieval_k = int((alert or {}).get("retrieval_k", 50))
        rerank_n = int((alert or {}).get("rerank_n", 10))
        max_return = int((alert or {}).get("max_articles_per_alert", 3))  # Default 3 articles per request

        # 4) Retrieve top-K by embeddings
        candidates = retrieve_candidates(recent, contextual_query, embedding_model, k=retrieval_k)

        # 5) Rerank with intent-aware scorer
        reranked = rerank_candidates(contextual_query, alert, candidates, embedding_model, n=rerank_n)

        # 5b) Neural Reranker: Advanced reranking with learned preferences
        neural_reranker = get_neural_reranker()
        if neural_reranker is not None:
            try:
                logger.info("🎯 Applying neural reranker with feedback learning...")
                reranked = neural_reranker.rerank_with_intelligence(
                    contextual_query, 
                    reranked, 
                    top_k=rerank_n
                )
            except Exception as e:
                logger.warning(f"Neural reranker skipped: {e}")
        
        # 5c) Optional cross-encoder rerank for stronger precision (fallback if neural fails)
        if neural_reranker is None:
            ce = get_ce_reranker()
            if ce is not None:
                try:
                    # Rerank top rerank_n items using cross-encoder
                    reranked = ce.rerank(contextual_query, reranked)
                except Exception as _:
                    pass
        
        # 5d) Personalization: Apply user-specific preferences
        personalization_engine = get_personalization_engine()
        if personalization_engine and user_id:
            try:
                logger.info(f"👤 Applying personalization for user {user_id}...")
                reranked = personalization_engine.personalize_articles(
                    user_id, 
                    reranked, 
                    blend_factor=0.6  # 60% personalization, 40% base ranking
                )
            except Exception as e:
                logger.warning(f"Personalization skipped: {e}")

        # 6) Diversify (MMR)
        diversified = mmr_diversify(reranked, k=max_return, alpha=0.7)

        # 7) Popularity/freshness fusion and final ordering with NLP quality
        fused = []
        min_fresh = float((alert or {}).get("min_fresh_weight", 0.25))
        for it in diversified:
            cat = (alert or {}).get("main_category", "") or category
            pop = popularity_proxy(it, cat)
            fresh = time_decay(it.get("published_date", ""), cat)
            # Enforce freshness floor if configured
            if fresh < min_fresh:
                continue
            
            # Enhanced preference core: neural > cross-encoder > rerank > retrieve
            pref = float(it.get("neural_rerank_score", 
                       it.get("blended_score",
                       it.get("ce_score", 
                       it.get("rerank_score", 
                       it.get("retrieve_score", 0.0))))))
            
            # Add NLP quality boost
            nlp_quality = 0.5
            if "nlp_intelligence" in it:
                nlp_quality = it["nlp_intelligence"].get("quality_score", 0.5)
            
            it2 = dict(it)
            it2["popularity_score"] = float(pop)
            it2["fresh_weight"] = float(fresh)
            it2["nlp_quality"] = float(nlp_quality)
            it2["preference_core"] = float(pref)
            
            # Compute final combined score
            # POPULARITY-FIRST STRATEGY: Popularity gets highest weight
            it2["final_score"] = (
                pop * 0.35 +             # Popularity (INCREASED to 35%)
                pref * 0.35 +            # AI ranking  
                fresh * 0.20 +           # Freshness
                nlp_quality * 0.10       # Content quality
            )
            
            # Remove internal vector to avoid JSON serialization issues
            if "_vec" in it2:
                try:
                    del it2["_vec"]
                except Exception:
                    it2["_vec"] = None
            fused.append(it2)

        # Sort by popularity FIRST (descending), then by final_score
        fused.sort(key=lambda x: (
            x.get("popularity_score", 0.0),  # Primary: Popularity
            x.get("final_score", 0.0)         # Secondary: AI score
        ), reverse=True)
        
        # Filter to top N most popular articles BEFORE Gemini (save API costs)
        # Gemini will only judge the most popular/trusted sources
        top_popular_count = min(10, len(fused))  # Top 10 most popular, or less if fewer available
        top_popular = fused[:top_popular_count]
        
        logger.info(f"📊 Popularity filter: {len(fused)} articles -> top {len(top_popular)} most popular")
        if top_popular:
            logger.info(f"   Top article: '{top_popular[0].get('title', '')[:60]}...' (pop: {top_popular[0].get('popularity_score', 0):.3f})")

        # 8) Gemini judge: Only judges top popular articles for final selection
        judge = get_gemini_judge()
        final = top_popular  # Default to top popular if Gemini fails
        if judge is not None:
            try:
                # Gemini picks best from top popular articles
                final = judge.judge(alert, top_popular, max_select=max_return)
                logger.info(f"✅ Gemini judge selected {len(final)} from top {len(top_popular)} popular articles")
            except Exception as e:
                logger.warning(f"Gemini judge skipped: {e}")
                final = top_popular
        
        logger.info(f"✅ AI Pipeline v3 Complete: {len(articles)} total -> {len(recent)} recent -> {len(candidates)} candidates -> {len(reranked)} reranked -> {len(diversified)} diversified -> {len(final)} final")
        
        # Log intelligence metrics
        if neural_reranker:
            metrics = neural_reranker.get_performance_metrics()
            logger.info(f"📊 Neural reranker metrics: {metrics.get('total_feedback', 0)} feedback, avg engagement: {metrics.get('avg_engagement', 0):.2f}")
        
        # Coerce any numpy scalar types to Python floats/ints for JSON safety
        safe = []
        for it in final[:max_return]:
            clean = {}
            for k, v in it.items():
                try:
                    if hasattr(v, "item") and callable(getattr(v, "item")):
                        clean[k] = v.item()
                    else:
                        clean[k] = v
                except Exception:
                    clean[k] = str(v)
            safe.append(clean)
        return safe
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return []
