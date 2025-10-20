"""
Article Filtering and Scoring Module
Handles relevance scoring, spam filtering, and intelligent content filtering
"""

import logging
import re
from typing import List, Dict
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import numpy as np
from urllib.parse import urlparse

from .config import (
    MIN_RELEVANCE_SCORE,
    MIN_KEYWORD_SCORE,
    MIN_SEMANTIC_SCORE,
    ARTICLE_TIME_WINDOW_HOURS,
    MAX_RELEVANT_ARTICLES,
    W_SEMANTIC,
    W_PERSONAL,
    W_BM25,
    W_POPULARITY,
    FRESH_TAU_BY_CATEGORY,
    SOURCE_AUTHORITY,
    GENDER_BONUS,
    POPULARITY_THRESHOLD,
    INTENT_PATTERNS,
    FORMAT_MATCH_BONUS,
    FORMAT_MISMATCH_PENALTY,
    HIGHLIGHTS_BONUS,
    FINAL_SCORES_BONUS,
    NEGATIVE_PENALTY,
    EVENT_BONUS,
    CATEGORY_QUERY_SYNONYMS,
    EXPLICIT_GENDER_BONUS,
)

logger = logging.getLogger(__name__)


def remove_obvious_spam_per_alert(articles: list, alert: dict) -> list:
    """Strong ML/RAG-based filtering - reduces dependency on Gemini"""
    try:
        if not articles or not alert:
            return articles

        filtered_articles = []
        keywords = [k.lower() for k in alert.get("sub_categories", [])]
        category = alert.get("main_category", "").lower()
        followup_questions = [q.lower() for q in alert.get("followup_questions", [])]
        custom_question = alert.get("custom_question", "").lower()

        # Build comprehensive user intent profile
        all_user_text = ' '.join(keywords + followup_questions + [custom_question])
        user_intent_keywords = set(word for word in all_user_text.split() if len(word) > 3)

        # If no user intent, accept all articles from same category
        if not user_intent_keywords:
            return articles

        for article in articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower() + ' ' + article.get('summary', '').lower() + ' ' + article.get('description', '').lower()
            article_text = title + ' ' + content
            article_category = article.get('category', '').lower()

            # STEP 1: Category must match
            if article_category != category:
                continue

            # STEP 2: ML/RAG Relevance Score
            relevance_score = article.get('relevance_score', 0)

            # STEP 3: Keyword overlap analysis
            matched_keywords = [kw for kw in user_intent_keywords if kw in article_text]
            keyword_match_count = len(matched_keywords)
            keyword_match_ratio = keyword_match_count / len(user_intent_keywords) if user_intent_keywords else 0

            # STEP 4: Title relevance
            title_keyword_matches = sum(1 for kw in user_intent_keywords if kw in title)

            # STEP 5: Relaxed ML-based filtering
            keep_article = False

            if relevance_score > 0.2:
                keep_article = True
            elif keyword_match_count >= 1:
                keep_article = True
            elif title_keyword_matches >= 1:
                keep_article = True
            elif keyword_match_ratio >= 0.1:
                keep_article = True

            if keep_article:
                # Add match metadata for Gemini to use
                article['ml_keyword_matches'] = matched_keywords
                article['ml_match_ratio'] = keyword_match_ratio
                article['ml_confidence'] = relevance_score
                filtered_articles.append(article)

        logger.info(f"ML/RAG filter for {category}: {len(articles)} -> {len(filtered_articles)} articles")
        return filtered_articles

    except Exception as e:
        logger.error(f"Error in ML/RAG filtering: {e}")
        return articles


def filter_recent_articles(articles: list, max_hours: int = ARTICLE_TIME_WINDOW_HOURS) -> list:
    """Filter articles newer than max_hours (default from config)."""
    now = datetime.now(timezone.utc)
    cutoff_time = now - timedelta(hours=max_hours)
    logger.info(f"📅 Current UTC time: {now} | Cutoff: {cutoff_time}")

    recent_articles = []
    for article in articles:
        try:
            published_date_str = article.get('published_date', '')
            article_title = article.get('title', '')[:50]

            if published_date_str:
                try:
                    published_date = parsedate_to_datetime(published_date_str)

                    if published_date >= cutoff_time:
                        recent_articles.append(article)
                        logger.debug(f"   ✅ Date OK: '{article_title}' | {published_date_str}")
                    else:
                        logger.debug(f"   ❌ Too old: '{article_title}' | {published_date_str}")
                except Exception as e:
                    # If date parsing fails, include the article
                    recent_articles.append(article)
                    logger.debug(f"   ⚠️  Date parse failed (INCLUDED): '{article_title}' | Error: {e}")
            else:
                # If no date, include the article
                recent_articles.append(article)
                logger.debug(f"   ℹ️  No date (INCLUDED): '{article_title}'")
        except Exception as e:
            recent_articles.append(article)
            logger.info(f"   ❗ Exception (INCLUDED): '{article_title}' | {e}")

    logger.info(f"⏰ Time filter: {len(articles)} -> {len(recent_articles)} recent articles")
    return recent_articles


def _time_decay(published_date_str: str, category: str) -> float:
    """Compute freshness weight using exponential decay based on category tau."""
    try:
        tau_hours = FRESH_TAU_BY_CATEGORY.get(category or "", 48)
        tau = max(float(tau_hours), 1.0)
        if not published_date_str:
            return 0.9  # neutral-ish if unknown
        published_dt = parsedate_to_datetime(published_date_str)
        age_hours = max((datetime.now(timezone.utc) - published_dt).total_seconds() / 3600.0, 0.0)
        return float(np.exp(-age_hours / tau))
    except Exception:
        return 0.8


def _source_authority_score(url: str) -> float:
    """Map article URL domain to an authority score using config."""
    try:
        if not url:
            return SOURCE_AUTHORITY.get("default", 0.7)
        netloc = urlparse(url).netloc.lower()
        # strip subdomain like www.
        parts = netloc.split(":")[0].split(".")
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
        # handle known domains that include subdomains like edition.cnn.com
        for known in SOURCE_AUTHORITY.keys():
            if known in netloc:
                return SOURCE_AUTHORITY[known]
        return SOURCE_AUTHORITY.get(domain, SOURCE_AUTHORITY.get("default", 0.7))
    except Exception:
        return SOURCE_AUTHORITY.get("default", 0.7)


def _compute_popularity_score(article: dict, category: str) -> float:
    """Simple popularity proxy combining source authority and recency.
    Placeholder until trending_stats is implemented."""
    authority = _source_authority_score(article.get("url", ""))
    fresh = _time_decay(article.get("published_date", ""), category)
    # Combine with higher weight on authority to avoid noise
    return float(0.6 * authority + 0.4 * fresh)


async def score_articles_with_embeddings(
    articles: list,
    alert_query: str,
    embedding_model,
    user_profile_vector: np.ndarray = None,
    popularity_mode: str = "boost",
    category: str = "",
    alert_followups: List[str] = None,
    custom_question: str = "",
) -> list:
    """Score articles using hybrid semantic + keyword + personalization + popularity.
    Backward-compatible: new params are optional.
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(alert_query)
        query_norm = np.linalg.norm(query_embedding) or 1.0

        # Normalize user profile vector if provided
        user_vec = None
        if user_profile_vector is not None and isinstance(user_profile_vector, np.ndarray):
            norm = np.linalg.norm(user_profile_vector)
            if norm > 0:
                user_vec = user_profile_vector / norm

        # Extract keywords from query
        query_keywords = set(re.findall(r'\b\w{4,}\b', alert_query.lower()))
        logger.info(f"🔍 Extracted {len(query_keywords)} keywords from query: {list(query_keywords)[:10]}")

        scored_articles = []

        # -------- Helper closures for intent parsing --------
        fu_list = [s.lower() for s in (alert_followups or [])]
        custom_lower = (custom_question or "").lower()

        def _format_pref() -> str | None:
            joined = " ".join(fu_list + [custom_lower])
            for fmt, patterns in INTENT_PATTERNS.get("formats", {}).items():
                if any(re.search(p, joined) for p in patterns):
                    return fmt
            return None

        def _prefer_highlights() -> bool:
            joined = " ".join(fu_list + [custom_lower])
            return any(re.search(p, joined) for p in INTENT_PATTERNS.get("intents", {}).get("prefer_highlights", []))

        def _want_final_scores() -> bool:
            joined = " ".join(fu_list + [custom_lower])
            return any(re.search(p, joined) for p in INTENT_PATTERNS.get("intents", {}).get("final_scores", []))

        def _has_negative_constraints(text: str) -> bool:
            if not (custom_lower or fu_list):
                return False
            joined = " ".join(fu_list + [custom_lower])
            wants_no_betting = any(re.search(p, joined) for p in INTENT_PATTERNS.get("intents", {}).get("no_betting", []))
            wants_no_gossip = any(re.search(p, joined) for p in INTENT_PATTERNS.get("intents", {}).get("no_gossip", []))
            if not (wants_no_betting or wants_no_gossip):
                return False
            tl = text.lower()
            betting_hit = any(re.search(p, tl) for p in INTENT_PATTERNS.get("negatives", {}).get("betting", [])) if wants_no_betting else False
            gossip_hit = any(re.search(p, tl) for p in INTENT_PATTERNS.get("negatives", {}).get("gossip", [])) if wants_no_gossip else False
            return betting_hit or gossip_hit

        def _infer_match_format(text: str) -> str:
            tl = text.lower()
            for fmt, patterns in INTENT_PATTERNS.get("formats", {}).items():
                if any(re.search(p, tl) for p in patterns):
                    return fmt
            return "unknown"

        def _is_highlights(text: str) -> bool:
            tl = text.lower()
            return any(re.search(p, tl) for p in INTENT_PATTERNS.get("positives", {}).get("highlights", []))

        def _is_major_event(text: str) -> bool:
            tl = text.lower()
            return any(re.search(p, tl) for p in INTENT_PATTERNS.get("events", []))

        def _user_gender_pref() -> str | None:
            joined = " ".join(fu_list + [custom_lower])
            if any(re.search(p, joined) for p in INTENT_PATTERNS.get("intents", {}).get("prefer_women", [])):
                return "women"
            if any(re.search(p, joined) for p in INTENT_PATTERNS.get("intents", {}).get("prefer_men", [])):
                return "men"
            return None

        def _infer_article_gender(text: str) -> str:
            tl = text.lower()
            genders = INTENT_PATTERNS.get("genders", {})
            for g, patterns in genders.items():
                if any(re.search(p, tl) for p in patterns):
                    return g
            return "unknown"

        fmt_pref = _format_pref()
        prefer_highlights = _prefer_highlights()
        want_final_scores = _want_final_scores()
        gender_pref = _user_gender_pref()

        for article in articles:
            article_text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}"
            article_embedding = embedding_model.encode(article_text)
            art_norm = np.linalg.norm(article_embedding) or 1.0

            # Calculate cosine similarity (semantic)
            similarity = float(np.dot(query_embedding, article_embedding) / (query_norm * art_norm))

            # Calculate keyword overlap score
            article_text_lower = article_text.lower()
            matched_keywords = [kw for kw in query_keywords if kw in article_text_lower]
            keyword_score = len(matched_keywords) / max(len(query_keywords), 1) if query_keywords else 0

            # Personalization: similarity to user profile
            personal_score = 0.0
            if user_vec is not None:
                personal_score = float(np.dot(user_vec, article_embedding / art_norm))

            # Popularity: authority + recency
            popularity_score = _compute_popularity_score(article, category)

            # Freshness multiplier using decay
            fresh_weight = _time_decay(article.get("published_date", ""), category)

            # Gender bonus when explicitly preferred by user and detected in article
            gender_bonus = 0.0
            if gender_pref:
                art_gender = _infer_article_gender(article_text)
                if art_gender == gender_pref:
                    gender_bonus += EXPLICIT_GENDER_BONUS

            # Intent-driven bonuses/penalties
            intent_bonus = 0.0
            intent_penalty = 0.0

            # Match format preference if specified
            if fmt_pref is not None:
                art_fmt = _infer_match_format(article_text)
                if art_fmt == fmt_pref:
                    intent_bonus += FORMAT_MATCH_BONUS
                elif art_fmt != "unknown":
                    intent_penalty += FORMAT_MISMATCH_PENALTY

            # Prefer highlights if requested
            if prefer_highlights and _is_highlights(article_text):
                intent_bonus += HIGHLIGHTS_BONUS

            # Include final scores if requested (detect simple score patterns)
            if want_final_scores and any(re.search(p, article_text) for p in INTENT_PATTERNS.get("positives", {}).get("final_scores", [])):
                intent_bonus += FINAL_SCORES_BONUS

            # Negative constraints (betting/gossip)
            if _has_negative_constraints(article_text):
                intent_penalty += NEGATIVE_PENALTY

            # Major event bonus
            if _is_major_event(article_text):
                intent_bonus += EVENT_BONUS

            # Blended base score
            blended = (
                W_SEMANTIC * similarity
                + W_PERSONAL * personal_score
                + W_BM25 * 0.0  # BM25 disabled by default
                + W_POPULARITY * popularity_score
            )

            final_score = float(blended * fresh_weight + gender_bonus + intent_bonus - intent_penalty)

            # Back-compat hybrid score (semantic + keyword)
            hybrid_score = (0.5 * similarity) + (0.5 * keyword_score)

            article['relevance_score'] = float(hybrid_score)
            article['semantic_score'] = float(similarity)
            article['keyword_score'] = float(keyword_score)
            article['matched_keywords'] = matched_keywords
            article['personal_score'] = float(personal_score)
            article['popularity_score'] = float(popularity_score)
            article['fresh_weight'] = float(fresh_weight)
            article['final_relevance_score'] = float(final_score)

            # Popularity-only mode: drop low popularity
            if popularity_mode == "only_popular" and popularity_score < POPULARITY_THRESHOLD:
                logger.info(f"   ❌ REJECTED (popularity): '{article.get('title', '')[:60]}' | Popularity:{popularity_score:.3f}")
                continue

            # Accept if any threshold is met (keep original thresholds for safety)
            if final_score > MIN_RELEVANCE_SCORE or keyword_score > MIN_KEYWORD_SCORE or similarity > MIN_SEMANTIC_SCORE:
                scored_articles.append(article)
                logger.info(
                    f"   ✅ MATCHED: '{article.get('title', '')[:60]}' | Final:{final_score:.3f} Sem:{similarity:.3f} Key:{keyword_score:.3f} Pers:{personal_score:.3f} Pop:{popularity_score:.3f} Fresh:{fresh_weight:.3f} Bonus:{intent_bonus:.3f} Pen:{intent_penalty:.3f}"
                )
            else:
                logger.info(
                    f"   ❌ REJECTED: '{article.get('title', '')[:60]}' | Final:{final_score:.3f} Sem:{similarity:.3f} Key:{keyword_score:.3f} Pers:{personal_score:.3f} Pop:{popularity_score:.3f} Fresh:{fresh_weight:.3f} Bonus:{intent_bonus:.3f} Pen:{intent_penalty:.3f}"
                )

        # Sort by final score fallback to relevance_score
        scored_articles.sort(key=lambda x: x.get('final_relevance_score', x.get('relevance_score', 0)), reverse=True)

        # Popularity diagnostics
        if scored_articles:
            try:
                pops = [float(a.get('popularity_score', 0.0)) for a in scored_articles]
                avg_pop = sum(pops) / max(len(pops), 1)
                logger.info(f"📈 Popularity summary: avg={avg_pop:.3f} max={max(pops):.3f} min={min(pops):.3f}")
                top_pop = sorted(scored_articles, key=lambda x: x.get('popularity_score', 0.0), reverse=True)[:5]
                for idx, a in enumerate(top_pop, 1):
                    logger.info(f"   🔝 Pop#{idx}: {a.get('popularity_score', 0.0):.3f} | '{a.get('title','')[:60]}'")
            except Exception as _:
                pass

        logger.info(f"🎯 Scoring complete: {len(articles)} -> {len(scored_articles)} relevant articles")
        return scored_articles[:MAX_RELEVANT_ARTICLES]

    except Exception as e:
        logger.error(f"Error scoring articles: {e}")
        return articles


async def build_contextual_query(alert: dict, keywords: list, followup_questions: list, custom_question: str, category: str) -> str:
    """Build a dynamic contextual query using data-driven synonyms and intents.
    - Adds category
    - Adds user keywords and followups
    - Enriches with per-category synonyms from config
    - Derives intent tokens (formats, highlights, final scores) via INTENT_PATTERNS
    """
    try:
        query_parts: List[str] = []
        _ = alert  # reserved

        cat = (category or '').lower().strip()
        if cat:
            query_parts.append(cat)

        # User-provided context
        for kw in (keywords or []):
            if kw and isinstance(kw, str):
                query_parts.append(kw)
        for fu in (followup_questions or []):
            if fu and isinstance(fu, str):
                query_parts.append(fu)

        # Per-category enrichment (data-driven)
        for syn in CATEGORY_QUERY_SYNONYMS.get(cat, []):
            query_parts.append(syn)

        # Intent enrichment (formats, highlights, final scores) from followups/custom
        joined_intent = " ".join([" ".join([k for k in (keywords or []) if isinstance(k, str)]),
                                   " ".join([f for f in (followup_questions or []) if isinstance(f, str)]),
                                   (custom_question or "")]).lower()

        # Formats
        for fmt, patterns in INTENT_PATTERNS.get('formats', {}).items():
            if any(re.search(p, joined_intent) for p in patterns):
                query_parts.append(fmt)

        # Prefer highlights
        if any(re.search(p, joined_intent) for p in INTENT_PATTERNS.get('intents', {}).get('prefer_highlights', [])):
            query_parts.append('highlights')

        # Final scores
        if any(re.search(p, joined_intent) for p in INTENT_PATTERNS.get('intents', {}).get('final_scores', [])):
            query_parts.append('final scores')

        # Gender preference token (to steer embeddings)
        if any(re.search(p, joined_intent) for p in INTENT_PATTERNS.get('intents', {}).get('prefer_women', [])):
            query_parts.append('women')
        elif any(re.search(p, joined_intent) for p in INTENT_PATTERNS.get('intents', {}).get('prefer_men', [])):
            query_parts.append('men')

        # General meaningful tokens from custom
        if custom_question and custom_question.strip():
            meaningful_words = [w for w in custom_question.split() if len(w) > 3]
            query_parts.extend(meaningful_words)

        contextual_query = " ".join(query_parts)
        logger.info(f"Built contextual query: {contextual_query}")
        return contextual_query

    except Exception as e:
        logger.error(f"Error building contextual query: {e}")
        simple_query = " ".join(([category] if category else []) + (keywords or []) + (followup_questions or []) + ([custom_question] if custom_question else []))
        return simple_query.strip()


async def debug_user_satisfaction(articles: list, user_alerts: list) -> dict:
    """Debug function to check if articles will satisfy user requirements"""
    try:
        if not articles or not user_alerts:
            return {"status": "no_data", "message": "No articles or alerts to check"}

        # Get user requirements
        user_requirements = {}
        for alert in user_alerts:
            user_requirements = {
                'main_category': alert.get('main_category', '').lower(),
                'keywords': [k.lower() for k in alert.get('sub_categories', [])],
                'followup_questions': [q.lower() for q in alert.get('followup_questions', [])],
                'custom_question': alert.get('custom_question', '').lower()
            }

        # Analyze each article
        satisfaction_analysis = []
        overall_satisfaction = 0

        for idx, article in enumerate(articles):
            title = article.get('title', '').lower()
            content = (article.get('content', '') + ' ' + article.get('summary', '')).lower()

            # Check satisfaction criteria
            matches = {
                'category_match': False,
                'keyword_match': False,
                'followup_match': False,
                'custom_match': False
            }

            # Category check
            user_category = user_requirements['main_category']
            article_category = article.get('category', '').lower()
            if user_category == article_category:
                matches['category_match'] = True

            # Keywords check
            if user_requirements['keywords']:
                for keyword in user_requirements['keywords']:
                    if keyword in title or keyword in content:
                        matches['keyword_match'] = True
                        break

            # Followup questions check
            if user_requirements['followup_questions']:
                for question in user_requirements['followup_questions']:
                    if question in title or question in content:
                        matches['followup_match'] = True
                        break

            # Custom question check
            if user_requirements['custom_question']:
                custom_words = user_requirements['custom_question'].split()
                if any(word in title or word in content for word in custom_words if len(word) > 3):
                    matches['custom_match'] = True

            # Calculate satisfaction score
            score = sum(matches.values()) / len(matches) * 100
            overall_satisfaction += score

            satisfaction_analysis.append({
                'article_id': idx,
                'title': article.get('title', '')[:50] + '...',
                'satisfaction_score': round(score, 1),
                'matches': matches,
                'has_image': bool(article.get('image_url', '')),
                'relevance_score': article.get('relevance_score', 0)
            })

        overall_satisfaction = overall_satisfaction / len(articles) if articles else 0

        # Determine satisfaction level
        if overall_satisfaction >= 75:
            satisfaction_level = "EXCELLENT - User will be very satisfied"
        elif overall_satisfaction >= 50:
            satisfaction_level = "GOOD - User will be satisfied"
        elif overall_satisfaction >= 25:
            satisfaction_level = "FAIR - User might be satisfied"
        else:
            satisfaction_level = "POOR - User may not be satisfied"

        debug_report = {
            'overall_satisfaction_score': round(overall_satisfaction, 1),
            'satisfaction_level': satisfaction_level,
            'user_requirements': user_requirements,
            'articles_analysis': satisfaction_analysis,
            'total_articles': len(articles),
            'recommendation': "These articles should satisfy user intent" if overall_satisfaction >= 50 else "Consider improving article relevance"
        }

        logger.info(f"Satisfaction Debug: {overall_satisfaction:.1f}% - {satisfaction_level}")
        return debug_report

    except Exception as e:
        logger.error(f"Error in satisfaction debug: {e}")
        return {"status": "error", "message": str(e)}
