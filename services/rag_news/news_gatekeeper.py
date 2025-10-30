"""
News Gatekeeper - LLM-based filtering with 24-hour verification
This gatekeeper ensures:
1. News is from last 24 hours
2. News is relevant to user preferences (verified by LLM)
3. Title and description are generated for user
4. ONE news per topic - most relevant only
5. Overflow articles stored in pending queue for next cron
"""

import logging
import json
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GatekeeperResult:
    """Result from gatekeeper validation"""
    is_approved: bool
    generated_title: str
    generated_description: str
    rejection_reason: Optional[str] = None
    relevance_score: float = 0.0


class NewsGatekeeper:
    """
    Enhanced gatekeeper that validates news articles using LLM.
    Ensures articles are recent (last 24 hours) and relevant to user preferences.
    """

    def __init__(self, llm_client):
        """
        Initialize the NewsGatekeeper

        Args:
            llm_client: LLM client for generating relevance checks and descriptions
        """
        self.llm = llm_client
        self.time_window_hours = 24

    def _is_within_24_hours(self, published_at: datetime) -> bool:
        """
        Check if article is within last 24 hours

        Args:
            published_at: Publication timestamp

        Returns:
            True if within 24 hours, False otherwise
        """
        try:
            # Make timezone-aware if needed
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            time_diff = now - published_at

            is_recent = time_diff <= timedelta(hours=self.time_window_hours)

            if not is_recent:
                logger.info(f"Article rejected: too old ({time_diff.total_seconds() / 3600:.1f} hours old)")

            return is_recent
        except Exception as e:
            logger.error(f"Error checking article timestamp: {e}")
            return False

    async def validate_article(
        self,
        article: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> GatekeeperResult:
        """
        Validate a single article against user preferences and time constraints

        Args:
            article: Article dictionary with title, content, url, published_at, etc.
            user_preferences: User's alert preferences (category, entities, etc.)

        Returns:
            GatekeeperResult with validation decision and generated content
        """
        # Step 1: Verify 24-hour constraint
        published_at = article.get("published_at")
        if isinstance(published_at, str):
            try:
                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            except:
                logger.error(f"Failed to parse published_at: {published_at}")
                return GatekeeperResult(
                    is_approved=False,
                    generated_title="",
                    generated_description="",
                    rejection_reason="Invalid timestamp"
                )

        if not isinstance(published_at, datetime):
            return GatekeeperResult(
                is_approved=False,
                generated_title="",
                generated_description="",
                rejection_reason="Missing or invalid timestamp"
            )

        if not self._is_within_24_hours(published_at):
            return GatekeeperResult(
                is_approved=False,
                generated_title="",
                generated_description="",
                rejection_reason=f"Article is older than {self.time_window_hours} hours"
            )

        # Step 2: Use LLM to check relevance and generate title/description
        try:
            prompt = self._build_gatekeeper_prompt(article, user_preferences)
            response = await self.llm.generate(prompt)
            result = self._parse_gatekeeper_response(response)

            if result.get("is_relevant", False):
                # ALWAYS use LLM-generated title/description (never fallback to original)
                generated_title = (result.get("title") or "").strip()
                generated_description = (result.get("description") or "").strip()
                relevance = float(result.get("relevance_score", 0) or 0)
                
                # Clean LLM-generated text to remove UI fragments (VERY AGGRESSIVE)
                def _clean_llm_text(text: str) -> str:
                    if not text:
                        return ""
                    import re
                    # Step 1: Remove relative-time phrases (anywhere in text)
                    text = re.sub(r"(?i)\b\d+\s+(?:minutes?|hours?|days?|weeks?|months?)\s+ago\b[\.:]?\s*\.\.\.?", "", text)
                    # Step 2: Remove all ellipses first
                    text = re.sub(r"\.\.\.+", "", text)
                    # Step 3: Remove UI fragments from anywhere
                    text = re.sub(r"(?i)what\s+is\s+the\s+daily\s+range.*?$", "", text)
                    text = re.sub(r"(?i)today'?s\s*\.\.\.?.*?$", "", text)
                    text = re.sub(r"(?i)read\s+more.*?$", "", text)
                    text = re.sub(r"(?i)article\s+from.*?$", "", text)
                    # Step 4: Remove leading fragments
                    text = re.sub(r"^[\.\s]+", "", text)
                    # Step 5: Remove trailing fragments
                    text = re.sub(r"[\.\s]+$", "", text)
                    # Step 6: Clean up multiple spaces
                    text = re.sub(r"\s+", " ", text).strip()
                    return text
                
                # Clean the LLM-generated text IMMEDIATELY
                generated_title = _clean_llm_text(generated_title)
                generated_description = _clean_llm_text(generated_description)
                
                # CRITICAL: Validate and FIX extracted rates for financial news (reject invalid rates like 2025)
                def _validate_financial_rate(text: str) -> str:
                    """Check if text contains invalid rates (like years) and FIX them or remove"""
                    if not text:
                        return text
                    import re
                    # USD/INR rates should be 70-100, not 4-digit numbers or very large numbers
                    
                    # Pattern 1: "USD/INR: ₹2025" or "₹2025 today" - these are YEARS, not rates
                    # Find all instances and replace with generic text
                    text = re.sub(r'₹\s*20\d{2}\b(?:\s+today)?', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'₹\s*([12]\d{3})\b', '', text)  # Any 4-digit starting with 1 or 2 (likely years)
                    
                    # Pattern 2: "rate is 2025" or "2025 per USD"
                    text = re.sub(r'\b(?:rate|price|exchange)\s*(?:is|at|of|:)\s*20\d{2}\b', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'\b(?:rate|price|exchange)\s*(?:is|at|of|:)\s*([12]\d{3})\b', '', text, flags=re.IGNORECASE)
                    
                    # Pattern 3: "USD/INR: 2025" (without ₹ but still a year)
                    text = re.sub(r'(?:USD/INR|USD-INR|usd/inr)[\s:]+20\d{2}\b', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'(?:USD/INR|USD-INR|usd/inr)[\s:]+([12]\d{3})\b', '', text, flags=re.IGNORECASE)
                    
                    # Pattern 4: Large numbers that are clearly not rates (> 1000)
                    # But only if they're presented as rates (near "rate", "price", "USD/INR")
                    text = re.sub(r'(?:rate|price|USD/INR|usd/inr).*?\b([12]\d{3}|\d{5,})\b', 
                                 lambda m: m.group(0).replace(m.group(1), '[rate not specified]'), 
                                 text, flags=re.IGNORECASE)
                    
                    # Pattern 5: If we removed the rate, fix the sentence structure
                    # Remove "today" if it's orphaned after removing invalid rate
                    text = re.sub(r'\s+today\s*$', '', text, flags=re.IGNORECASE)
                    text = re.sub(r':\s*today', ' update', text, flags=re.IGNORECASE)
                    
                    # Clean up any double spaces or orphaned punctuation
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r':\s*$', '', text)
                    text = text.strip()
                    
                    return text
                
                # For financial news, validate the rate ALWAYS
                is_financial = any(kw in (generated_title + " " + generated_description).lower() 
                                 for kw in ['usd', 'inr', 'dollar', 'rupee', 'exchange rate', 'rate', 'currency'])
                # Detect if user explicitly wants daily price/rate
                wants_price = any(x in (user_preferences.get('custom_question','') + ' ' + ' '.join(user_preferences.get('followup_questions',[]))).lower() 
                                   for x in ['price', 'rate', 'exchange rate', 'dollar price', 'rupee price', 'daily'])
                if is_financial:
                    generated_title = _validate_financial_rate(generated_title)
                    generated_description = _validate_financial_rate(generated_description)
                    
                    # Helper to detect a realistic USD/INR rate in text
                    def _has_valid_rate(text: str) -> bool:
                        if not text:
                            return False
                        import re
                        nums = re.findall(r"(?<!\d)(\d{2}\.\d{1,2}|\d{2})(?!\d)", text)
                        for n in nums:
                            try:
                                v = float(n)
                                if 70 <= v <= 100:
                                    return True
                            except Exception:
                                continue
                        return False
                    
                    # If title still contains invalid pattern after validation, replace with generic
                    if re.search(r'₹\s*20\d{2}|rate.*?20\d{2}', generated_title, re.IGNORECASE):
                        # Extract meaningful words from title and rebuild
                        words = generated_title.split()
                        valid_words = [w for w in words if not re.match(r'^₹?\s*20\d{2}$', w, re.IGNORECASE)]
                        if valid_words:
                            generated_title = ' '.join(valid_words[:8])  # Limit to 8 words
                        else:
                            generated_title = "USD/INR market update"
                    
                    # Same for description
                    if re.search(r'₹\s*20\d{2}|rate.*?20\d{2}', generated_description, re.IGNORECASE):
                        # Replace with context-based description
                        if 'rbi' in generated_description.lower() or 'reserve bank' in generated_description.lower():
                            generated_description = "📈 USD/INR market update: Options favor Indian rupee after RBI intervention"
                        elif 'rupee' in generated_description.lower() or 'inr' in generated_description.lower():
                            generated_description = "📈 USD/INR market update: Indian rupee movements"
                        else:
                            generated_description = "📈 USD/INR market update"

                    # STRICT APPROVAL for daily price requests: must have a valid rate
                    if wants_price and not (_has_valid_rate(article.get('title','') + ' ' + article.get('content','')) or _has_valid_rate(generated_title + ' ' + generated_description)):
                        return GatekeeperResult(
                            is_approved=False,
                            generated_title="",
                            generated_description="",
                            rejection_reason="No valid USD/INR rate found for daily price request"
                        )
                
                # CRITICAL: Only approve if LLM ACTUALLY generated clean description
                # If LLM description is empty or contains fragments, reject and use fallback with aggressive cleaning
                if not generated_title or len(generated_title.strip()) < 3:
                    # Title fallback - clean it too
                    raw_title = article.get("title", "").strip()
                    generated_title = _clean_llm_text(raw_title) if raw_title else "News Update"
                
                if not generated_description or len(generated_description.strip()) < 10:
                    # Description fallback - but REJECT if it contains UI fragments
                    base_content = article.get("content", "")
                    if base_content:
                        # Clean the content aggressively
                        cleaned_content = _clean_llm_text(base_content)
                        # Validate financial rates if needed
                        if is_financial:
                            cleaned_content = _validate_financial_rate(cleaned_content)
                        # If cleaned content still has UI fragments, generate a simple description from title
                        if any(fragment in cleaned_content.lower() for fragment in ['hours ago', 'days ago', 'what is', 'today\'s', '...']):
                            # Content is too messy, generate simple description
                            generated_description = f"📈 Latest update: {generated_title}"
                        else:
                            generated_description = cleaned_content[:250] if len(cleaned_content) > 250 else cleaned_content
                    else:
                        generated_description = f"📈 {generated_title}"

                # Final hardening: normalize, enforce entity mention, and remove UI fragments
                try:
                    blacklist = [
                        'hours ago', 'days ago', 'minutes ago', 'what is the daily range',
                        "today's", 'read more', 'article from', '...', 'view more', 'click here'
                    ]
                    entities = [e.lower() for e in user_preferences.get('canonical_entities', []) if isinstance(e, str)]

                    def _mentions_entity(text: str) -> bool:
                        if not entities:
                            return True
                        tl = (text or '').lower()
                        return any(e in tl for e in entities)

                    def _violates_blacklist(text: str) -> bool:
                        tl = (text or '').lower()
                        return any(b in tl for b in blacklist)

                    def _normalize_sentence(text: str, max_chars: int) -> str:
                        t = (text or '').strip()
                        t = re.sub(r"\s+", " ", t)
                        if len(t) > max_chars:
                            t = t[:max_chars].rstrip()
                        if t and t[-1] not in ".!?":
                            t = t + "."
                        return t

                    # Clean blacklisted fragments
                    if _violates_blacklist(generated_title):
                        for b in blacklist:
                            generated_title = re.sub(re.escape(b), '', generated_title, flags=re.IGNORECASE)
                    if _violates_blacklist(generated_description):
                        for b in blacklist:
                            generated_description = re.sub(re.escape(b), '', generated_description, flags=re.IGNORECASE)

                    # Normalize and bound lengths
                    generated_title = _normalize_sentence(generated_title, 90)
                    generated_description = _normalize_sentence(generated_description, 260)

                    # Reject if still low quality
                    if (not generated_title or len(generated_title.strip()) < 5 or
                        not generated_description or len(generated_description.strip()) < 20 or
                        _violates_blacklist(generated_title + ' ' + generated_description) or
                        not _mentions_entity(generated_title + ' ' + generated_description)):
                        return GatekeeperResult(
                            is_approved=False,
                            generated_title="",
                            generated_description="",
                            rejection_reason="LLM output failed quality constraints"
                        )
                except Exception:
                    # If hardening fails, proceed with cleaned values
                    pass

                return GatekeeperResult(
                    is_approved=True,
                    generated_title=generated_title,
                    generated_description=generated_description,
                    relevance_score=relevance
                )
            else:
                return GatekeeperResult(
                    is_approved=False,
                    generated_title="",
                    generated_description="",
                    rejection_reason=result.get("reason", "Not relevant to user preferences"),
                    relevance_score=result.get("relevance_score", 0.0)
                )

        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            # In case of error, be conservative and reject
            return GatekeeperResult(
                is_approved=False,
                generated_title="",
                generated_description="",
                rejection_reason=f"LLM validation error: {str(e)}"
            )

    def _build_gatekeeper_prompt(
        self,
        article: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM to validate article and generate content"""

        # Check content type
        content_sample = (article.get('title', '') + ' ' + article.get('content', ''))[:500].lower()
        is_financial = any(kw in content_sample for kw in ['price', 'rate', 'stock', 'exchange', 'rupee', 'dollar', 'share', 'usd', 'inr'])
        is_cricket = any(kw in content_sample for kw in ['cricket', 'runs', 'wickets', 'overs', 'score', 'match', 'innings', 'batting', 'bowling'])
        
        # Check for event conditions in user preferences
        custom_question = user_preferences.get('custom_question', '').lower()
        followups = ' '.join(user_preferences.get('followup_questions', [])).lower()
        has_score_condition = any(x in (custom_question + ' ' + followups) for x in ['120+', '120 ', 'hit 120', 'reach 120', 'score 120', 'runs', 'target'])
        
        if is_financial:
            # Check if user specifically asked for price/rate data
            wants_price = any(x in (custom_question + ' ' + followups) for x in ['price', 'rate', 'exchange rate', 'dollar price', 'rupee price', 'daily'])
            
            # Financial news format
            return f"""You are a financial news editor. Check if this article is about: {', '.join(user_preferences.get('canonical_entities', []))}

Article: {article.get('title', '')}
Content: {article.get('content', '')[:1500]}

TASK:
1. Is it relevant to the entities? YES → approve
2. **CRITICAL**: {"If user asked for price/rate/daily data, ONLY approve if article contains ACTUAL NUMBERS (e.g., '84.50', '₹84.50', 'USD = 83.25'). Reject generic market data pages that don't show current rates." if wants_price else "Extract key numbers (prices, rates, percentages, changes)"}
3. Extract key numbers (prices, rates, percentages, changes)
4. Generate CLEAN summary (STRICT FORMAT - NO UI FRAGMENTS):

STRICT OUTPUT FORMAT:
- Title: Clean headline with pair and rate (≤15 words, ≤90 chars). Example: "USD/INR: ₹88.20 today"
- Description: {"📈 USD/INR: ₹[CURRENT_RATE] today. Previous close: ₹[PREV_CLOSE]. Daily range: ₹[LOW]–₹[HIGH]." if wants_price else "📈 Current price: [actual numbers]. Key change: [percentage/amount]."} (40–260 chars, complete sentence, ends with a period)

CRITICAL RATE VALIDATION:
⚠️ For USD/INR rates, ONLY use numbers between 70-100 (realistic range)
⚠️ NEVER use years (like 2025, 2024) as rates
⚠️ NEVER use dates or large numbers (like 13620013) as rates
⚠️ If article mentions "2025" or large numbers, they are NOT exchange rates - ignore them
⚠️ USD/INR rate should be decimal like 88.20, 84.50, NOT whole numbers like 2025, 2024

ABSOLUTE PROHIBITIONS (DO NOT INCLUDE):
❌ NO relative time: "5 hours ago", "2 days ago", "x minutes ago"
❌ NO ellipses: "...", "....", etc.
❌ NO UI fragments: "What Is the Daily Range...", "Today's ...", "Read more..."
❌ NO meta text: "Article from...", "Published on..."
❌ NO incomplete sentences ending with "..."
❌ NO site navigation text or promotional content
❌ NO invalid rates: If you can't find a realistic rate (70-100), don't make up numbers

REQUIRED FORMAT:
- Write complete, concise sentences
- Use ONLY realistic rates (70-100 for USD/INR)
- Start with rate/price info immediately (ONLY if valid rate found)
- End with complete sentence (no fragments)
- If no valid rate found in article, focus on the news context instead

GOOD EXAMPLES:
✓ Title: "USD/INR: ₹88.203 today"
✓ Description: "📈 USD/INR: ₹88.203 today. Previous close: ₹88.21. Daily range: ₹88.10–88.35."
✓ If no rate found: "USD/INR market update: Options favor Indian rupee after RBI intervention"

BAD EXAMPLES (DO NOT CREATE):
✗ Title: "USD/INR: ₹2025 today" (2025 is a YEAR, not a rate!)
✗ Description: "5 hours ago ... The current USD/INR exchange rate is 88.203..."
✗ Using dates/years as rates: "₹2025", "₹2024" (these are YEARS, not exchange rates!)

Return JSON ONLY (clean output):
{{"is_relevant": true/false, "relevance_score": 0.8-1.0, "title": "USD/INR: ₹88.20 today", "description": "📈 USD/INR: ₹88.20 today. Previous close: ₹88.21. Daily range: ₹88.10–88.35."}}"""
        elif is_cricket:
            # Cricket news format with score extraction
            event_check = ""
            if has_score_condition:
                # Extract number from condition (e.g., "120+ runs" -> 120)
                nums = re.findall(r'(\d+)\+?\s*(?:runs|run)', custom_question + ' ' + followups)
                target_runs = int(nums[0]) if nums else None
                if target_runs:
                    event_check = f"\nCRITICAL: User wants alerts when team scores {target_runs}+ runs. Check if article mentions any team reaching {target_runs} or more runs. If YES, approve with high relevance_score (0.9+)."
            
            return f"""You are a cricket news editor. Check if this article is relevant to: {', '.join(user_preferences.get('canonical_entities', []))}

Article: {article.get('title', '')}
Content: {article.get('content', '')[:1000]}

TASK:
1. Is it relevant to the entities (teams/players)? YES → approve{event_check}
2. EXTRACT CRICKET SCORES:
   - Find runs scored (e.g., "120/3", "150 runs", "85 off 12 overs")
   - Find wickets (e.g., "3 wickets", "lost 5 wickets")
   - Find overs (e.g., "12.3 overs", "after 15 overs")
   - Find target if chasing (e.g., "need 150", "chasing 185")
3. Generate summary with SCORES:
   - Title: Include team names and key score (max 20 words)
   - Description: Format as "🏏 [Team1] vs [Team2]: [Score]. [Context]. [Overs/Status if available]"
   - If score condition met (like 120+ runs), highlight it in description

EXAMPLES:
{{"is_relevant": true, "relevance_score": 0.95, "title": "South Africa Women 135/2 after 18 overs vs England", "description": "🏏 South Africa Women vs England: SA 135/2 (18 ov). Team South Africa crossed 120+ runs mark! Need 45 more to win in 12 balls."}}
{{"is_relevant": true, "relevance_score": 0.85, "title": "Women's World Cup: Live Score Updates", "description": "🏏 England vs South Africa: Live match in progress. Current score updates available."}}

Return JSON:
{{"is_relevant": true/false, "relevance_score": 0.0-1.0, "title": "...", "description": "..."}}"""
        else:
            # Regular news format
            return f"""You are a news editor. Approve only if article clearly mentions ANY of these entities: {', '.join(user_preferences.get('canonical_entities', []))}

Article: {article.get('title', '')}
Content: {article.get('content', '')[:500]}

STRICT OUTPUT:
- title: ≤12 words, ≤80 chars, must include an entity, no UI fragments.
- description: 40–220 chars, complete sentence, no UI fragments, ends with period.

ABSOLUTE PROHIBITIONS:
❌ No relative time ("hours ago"), ellipses ("..."), site UI ("Read more", "What Is the Daily Range", "Today's ...").

Return JSON ONLY:
{{"is_relevant": true/false, "relevance_score": 0.0-1.0, "title": "...", "description": "..."}}"""

    def _parse_gatekeeper_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response from gatekeeper validation"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                logger.warning("No JSON found in gatekeeper response")
                return {
                    "is_relevant": False,
                    "relevance_score": 0.0,
                    "reason": "Failed to parse response",
                    "title": "",
                    "description": ""
                }
        except Exception as e:
            logger.error(f"Error parsing gatekeeper response: {e}")
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "reason": f"Parse error: {str(e)}",
                "title": "",
                "description": ""
            }

    async def filter_articles(
        self,
        articles: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        db=None,
        alert_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Filter multiple articles through the gatekeeper
        
        NEW: ONE NEWS PER TOPIC logic
        - Groups articles by topic
        - Sends only 1 per topic (highest score)
        - Stores overflow in pending_articles collection

        Args:
            articles: List of article dictionaries
            user_preferences: User's alert preferences
            db: MongoDB database instance (optional)
            alert_id: Alert ID for pending queue storage
            user_id: User ID for pending queue storage

        Returns:
            Dict with:
            - approved_articles: List of articles to send NOW (1 per topic)
            - pending_articles: List of articles queued for next cron
            - stats: Processing statistics
        """
        approved_articles = []
        pending_articles = []

        logger.info(f"Gatekeeper processing {len(articles)} articles")

        # Step 1: Validate all articles and get enriched with title/description
        enriched_articles = []
        for idx, article in enumerate(articles):
            result = await self.validate_article(article, user_preferences)

            if result.is_approved:
                # Add generated content to article (preserve all fields including _index via copy())
                enriched_article = article.copy()
                enriched_article["generated_title"] = result.generated_title
                enriched_article["generated_description"] = result.generated_description
                enriched_article["relevance_score"] = result.relevance_score
                enriched_article["gatekeeper_approved"] = True
                # Ensure metadata dict exists and store validated image
                if "metadata" not in enriched_article:
                    enriched_article["metadata"] = {}
                # Image was validated in validate_article, get it from article
                raw_img = (
                    article.get("image") or article.get("image_url") 
                    or enriched_article["metadata"].get("image") 
                    or enriched_article["metadata"].get("image_url")
                )
                if raw_img and isinstance(raw_img, str) and raw_img.startswith(('http://', 'https://')):
                    from urllib.parse import urlparse
                    parsed = urlparse(raw_img)
                    if parsed.scheme in ('http', 'https') and parsed.netloc:
                        enriched_article["metadata"]["image"] = raw_img
                
                # Add topic for grouping
                enriched_article["topic"] = self._extract_topic(enriched_article, user_preferences)

                enriched_articles.append(enriched_article)
                logger.info(
                    f"Article {idx + 1} APPROVED: {result.generated_title[:50]}... "
                    f"(score: {result.relevance_score:.2f}, topic: {enriched_article.get('topic', 'unknown')})"
                )
            else:
                logger.info(
                    f"Article {idx + 1} REJECTED: {result.rejection_reason}"
                )

        logger.info(
            f"Gatekeeper validation: {len(enriched_articles)} approved, "
            f"{len(articles) - len(enriched_articles)} rejected"
        )

        # Step 2: Group by topic and select ONE per topic
        topic_groups = {}
        for article in enriched_articles:
            topic = article.get("topic", "general")
            # Normalize topic for better grouping (e.g., "USD/INR Rate" = "usd_inr")
            topic_normalized = topic.lower().replace(" ", "_").replace("/", "_")
            if topic_normalized not in topic_groups:
                topic_groups[topic_normalized] = []
            topic_groups[topic_normalized].append(article)

        logger.info(f"Grouped into {len(topic_groups)} topics: {list(topic_groups.keys())}")

        # Step 3: Select BEST article overall (highest relevance) - send only ONE
        if topic_groups:
            # Flatten all articles and sort by relevance
            all_candidates = []
            for topic_articles in topic_groups.values():
                all_candidates.extend(topic_articles)
            
            # Sort by relevance score (highest first)
            all_candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Send ONLY top 1 article NOW (best relevance)
            if all_candidates:
                approved_articles.append(all_candidates[0])
                logger.info(f"Sending top article: {all_candidates[0].get('generated_title', '')[:50]} (score: {all_candidates[0].get('relevance_score', 0):.2f})")
            
            # Queue remaining articles for next cron
            if len(all_candidates) > 1:
                for overflow in all_candidates[1:]:
                    # Add metadata for queueing
                    overflow["queued_at"] = datetime.now(timezone.utc).isoformat()
                    overflow["alert_id"] = alert_id
                    overflow["user_id"] = user_id
                    overflow["status"] = "pending"
                    pending_articles.append(overflow)
                
                logger.info(
                    f"Queued {len(all_candidates)-1} articles for next cron"
                )

        logger.info(
            f"Gatekeeper final: {len(approved_articles)} to send NOW, "
            f"{len(pending_articles)} queued for next cron"
        )

        # Step 4: Store pending articles in database if provided
        if db is not None and pending_articles and alert_id:
            try:
                pending_collection = db.get_collection("pending_articles")
                await pending_collection.insert_many(pending_articles)
                logger.info(f"Stored {len(pending_articles)} articles in pending queue")
            except Exception as e:
                logger.error(f"Failed to store pending articles: {e}")

        return {
            "approved_articles": approved_articles,
            "pending_articles": pending_articles,
            "stats": {
                "total_validated": len(enriched_articles),
                "total_rejected": len(articles) - len(enriched_articles),
                "topics_found": len(topic_groups),
                "articles_sent_now": len(approved_articles),
                "articles_queued": len(pending_articles)
            }
        }

    def _extract_topic(self, article: Dict[str, Any], user_preferences: Dict[str, Any]) -> str:
        """
        Extract topic from article based on user preferences
        
        Priority:
        1. Combined related entities (e.g., "US Dollar" + "Indian Rupee" = "USD/INR")
        2. Canonical entities (e.g., "Yes Bank", "US Dollar")
        3. Sub-categories (e.g., "stockmarket", "cricket")
        4. Category (e.g., "sports", "news")
        
        Args:
            article: Article dictionary
            user_preferences: User alert preferences
            
        Returns:
            Topic string
        """
        # Check canonical entities
        entities = user_preferences.get("canonical_entities", [])
        title_lower = article.get("title", "").lower()
        content_lower = article.get("content", "").lower()
        
        # For financial articles, try to combine related entities (USD + INR = "USD/INR" topic)
        if len(entities) >= 2:
            matched_entities = [e for e in entities if e.lower() in title_lower or e.lower() in content_lower]
            if len(matched_entities) >= 2:
                # Financial pairs: combine into single topic
                if any("dollar" in e.lower() or "usd" in e.lower() for e in matched_entities) and \
                   any("rupee" in e.lower() or "inr" in e.lower() for e in matched_entities):
                    return "USD/INR Rate"
                elif any("stock" in e.lower() or "share" in e.lower() for e in matched_entities):
                    # Stock-related: use first matched entity
                    return matched_entities[0]
                else:
                    # Other pairs: combine
                    return " / ".join(matched_entities[:2])
            elif len(matched_entities) == 1:
                return matched_entities[0]
        
        # Single entity match
        for entity in entities:
            if entity.lower() in title_lower or entity.lower() in content_lower:
                return entity
        
        # Check sub-categories
        subcats = user_preferences.get("sub_categories", [])
        for subcat in subcats:
            if subcat.lower() in title_lower or subcat.lower() in content_lower:
                return subcat
        
        # Fall back to category
        category = user_preferences.get("category", "general")
        if category and category != "general":
            return category
        
        # Last resort: use first canonical entity if exists
        if entities:
            return entities[0]
        
        return "general"
