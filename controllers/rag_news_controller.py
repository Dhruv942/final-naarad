from fastapi import APIRouter
from typing import Any, Dict, List
import logging

from services.rag_news.pipeline import run_pipeline
from services.rag_news.personalization import (
    build_user_profile,
    rerank_articles,
    enforce_must_haves,
)
from services.rag_news.embeddings import get_embedder
from services.rag_news.article_filter import build_contextual_query
from services.rag_news.rss_fetcher import fetch_category_articles
from db.mongo import alerts_collection, db
from urllib.parse import urlparse, parse_qsl, urlencode
import hashlib
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/news/{user_id}")
async def get_user_news(user_id: str):
    try:
        alerts = await alerts_collection.find({"user_id": user_id, "is_active": True}).to_list(None)
        prefs = []
        results = []
        embedder = get_embedder()
        
        def _normalize_url(u: str) -> str:
            try:
                if not u:
                    return ""
                p = urlparse(u)
                host = p.netloc.lower()
                if host.startswith("www."):
                    host = host[4:]
                if host.startswith("m."):
                    host = host[2:]
                path = (p.path or "").rstrip("/").lower()
                keep = {"id", "p", "v", "storyid", "article", "amp"}
                qs = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True) if k.lower() in keep]
                query = urlencode(qs)
                key = f"{host}{path}"
                if query:
                    key = f"{key}?{query}"
                return key
            except Exception:
                return u or ""

        def _fingerprint(u: str, title: str) -> str:
            d = (urlparse(u).netloc or "").lower()
            if d.startswith("www."):
                d = d[4:]
            if d.startswith("m."):
                d = d[2:]
            tl = (title or "").strip().lower()
            return hashlib.sha1(f"{d}|{tl}".encode("utf-8", errors="ignore")).hexdigest()

        async def _already_sent(uid: str, url: str, title: str, dedup_hours: int = 48) -> bool:
            try:
                coll = db.get_collection("sent_notifications")
                nurl = _normalize_url(url)
                tl = (title or "").strip().lower()
                fp = _fingerprint(url, title)
                cutoff = datetime.now(timezone.utc) - timedelta(hours=dedup_hours)
                cnt = await coll.count_documents({
                    "user_id": uid,
                    "created_at": {"$gte": cutoff.isoformat()},
                    "$or": [
                        {"url": nurl},
                        {"url": url},
                        {"title_lower": tl},
                        {"fp": fp},
                    ],
                })
                return cnt > 0
            except Exception:
                return False

        async def _mark_sent(uid: str, url: str, title: str, alert_id: str):
            try:
                coll = db.get_collection("sent_notifications")
                nurl = _normalize_url(url)
                tl = (title or "").strip().lower()
                fp = _fingerprint(url, title)
                doc = {
                    "user_id": uid,
                    "url": nurl,
                    "raw_url": url,
                    "title": title,
                    "title_lower": tl,
                    "fp": fp,
                    "alert_id": alert_id,
                    "channel": "api",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                await coll.update_one({"user_id": uid, "url": nurl}, {"$set": doc}, upsert=True)
            except Exception as e:
                logger.warning(f"Mark sent failed: {e}")

        for a in alerts:
            cat = (a.get("main_category", "") or "").lower()
            kws = a.get("sub_categories", []) or []
            fus = a.get("followup_questions", []) or []
            cq = a.get("custom_question", "") or ""
            ctx_q = await build_contextual_query(a, kws, fus, cq, cat)
            prefs.append({
                "alert_id": str(a.get("_id")),
                "category": cat,
                "sub_categories": kws,
                "followup_questions": fus,
                "custom_question": cq,
                "contextual_query": ctx_q,
            })

            # Fetch articles from RSS + optionally Google Search
            # Pass keywords to enable targeted search
            search_keywords = kws + fus  # Combine sub_categories + followup_questions
            articles = await fetch_category_articles(cat, keywords=search_keywords)
            ranked = await run_pipeline(a, articles, embedder)
            # Personalization: build profile from alert and rerank
            profile = build_user_profile(a, user_id)
            personalized = rerank_articles(ranked, profile, ctx_q)
            personalized = enforce_must_haves(personalized, profile)
            
            # Persistent dedup: filter out items already sent to this user within dedup_hours window
            aid = str(a.get("_id"))
            dedup_hours = int(a.get("dedup_hours", 48))
            not_sent = []
            deduped_count = 0
            for it in personalized:
                url = it.get("url", "")
                title = it.get("title", "")
                if not await _already_sent(user_id, url, title, dedup_hours):
                    not_sent.append(it)
                else:
                    deduped_count += 1
                    logger.debug(f"   ⏭️  Deduped (already sent): {title[:50]}...")
            
            if deduped_count > 0:
                logger.info(f"⏭️  Deduplicated {deduped_count} articles (already sent within {dedup_hours}h)")
            
            # In-response dedup by normalized URL, raw URL, title_lower, and fingerprint
            seen = set()
            unique_ranked = []
            for it in not_sent:
                url = it.get("url", "")
                tl = (it.get("title", "") or "").strip().lower()
                nurl = _normalize_url(url)
                fp = _fingerprint(url, tl)
                keys = [f"u:{url}", f"n:{nurl}", f"t:{tl}", f"f:{fp}"]
                if any(k in seen for k in keys):
                    continue
                for k in keys:
                    seen.add(k)
                # annotate
                it2 = dict(it)
                it2["title_lower"] = tl
                it2["url_normalized"] = nurl
                it2["fp"] = fp
                unique_ranked.append(it2)
            
            # Log which articles are from Google Search
            google_count = sum(1 for art in unique_ranked if art.get("from_google_search", False))
            rss_count = len(unique_ranked) - google_count
            
            if len(unique_ranked) == 0:
                logger.warning(f"⚠️  Final Selection: 0 articles (all {len(personalized)} articles were deduplicated or filtered)")
            elif google_count > 0:
                logger.info(f"📊 Final Selection: {len(unique_ranked)} articles ({rss_count} RSS, {google_count} Google Search)")
                logger.info("🔍 Google Search articles in final results:")
                for i, art in enumerate(unique_ranked, 1):
                    if art.get("from_google_search", False):
                        logger.info(f"   {i}. {art.get('title', '')[:60]}... (Score: {art.get('final_score', 0):.3f})")
            else:
                logger.info(f"📊 Final Selection: {len(unique_ranked)} articles (all from RSS)")
            
            # Mark returned items as sent for future dedup
            for it in unique_ranked:
                await _mark_sent(user_id, it.get("url", ""), it.get("title", ""), aid)
            results.append({
                "alert_id": str(a.get("_id")),
                "category": cat,
                "contextual_query": ctx_q,
                "returned": len(unique_ranked),
                "articles": unique_ranked,
            })

        return {
            "status": "success",
            "user_id": user_id,
            "alerts_count": len(alerts),
            "preferences": prefs,
            "alert_results": results,
        }
    except Exception as e:
        logger.error(f"Failed to load user news prefs: {e}")
        return {"status": "error", "message": str(e)}
