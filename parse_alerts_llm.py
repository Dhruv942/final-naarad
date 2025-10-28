import asyncio
import json
import google.generativeai as genai
from db.mongo import client, alerts_collection
from bson import ObjectId
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Gemini API key from config
from services.rag_news.config import GEMINI_API_KEY
from services.rag_news.config import build_trusted_query, get_trusted_sport_sources

# Configure Gemini with user-provided key
genai.configure(api_key='AIzaSyBJqkrD1MteQ9FV6v3Dtdo39dhLUf4BRB4')

async def generate_alert_fields(alert_content, predefined_tags=None):
    """Generate comprehensive alert fields using Google's Gemini model with rule-based orchestration.

    predefined_tags: Optional flat list of tags to select from. If provided, the model
    must pick 1-5 matching tags into `tags` and, if none fit well, propose `custom_tags`.
    """
    try:
        # Parse alert content to extract user preferences
        alert_data = json.loads(alert_content) if isinstance(alert_content, str) else alert_content
        
        user_preferences = {
            "alert_id": str(alert_data.get("_id", alert_data.get("alert_id", ""))),
            "user_id": alert_data.get("user_id", ""),
            "category": alert_data.get("main_category", ""),
            "sub_categories": alert_data.get("sub_categories", []),
            "followup_questions": alert_data.get("followup_questions", []),
            "custom_question": alert_data.get("custom_question", "")
        }
        
        # Initialize the Gemini model per user request
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare the comprehensive prompt without f-string conflicts for literal JSON braces
        predefined_tags_json = json.dumps(predefined_tags or [], ensure_ascii=False, indent=2)

        head = (
            f"You are an Intelligent News Personalization Orchestrator for a RAG-driven alert system.\n"
            f"Your job: transform user preferences into a deep, rule-based structured alert profile\n"
            f"that will drive retrieval and filtering. No ML — only LLM + RAG.\n\n"
            f"Input:\n{json.dumps(user_preferences, indent=2)}\n\n"
            "Core Objectives:\n"
            "1️⃣ Understand true intent (not just keywords)\n"
            "2️⃣ Create event-driven filters (e.g., \"team must win\")\n"
            "3️⃣ Expand search intelligently (synonyms, related concepts)\n"
            "4️⃣ Define filtering logic (what to include + exclude)\n"
            "5️⃣ Output structured JSON following exact schema\n"
            "6️⃣ Generate a generic category-based Google News RSS URL ONLY\n\n"
            "7️⃣ Tagging: From the predefined tags list (if any), select 1-5 that best match the alert. If none fit, propose custom, concise and reusable tags.\n\n"
            "------------------------------------\n"
            "MANDATORY STEP FOR NEWS SOURCE URL\n"
            "------------------------------------\n"
            "✅ Always generate: google_rss_url based purely on the MAIN category only  \n"
            "✅ Do NOT include entities, teams, players, or tournament names in URL  \n"
            "✅ Never include followup_questions or custom keywords in URL  \n"
            "✅ Example format:\n\"https://news.google.com/rss/search?q=<category>&hl=en-IN&gl=IN&ceid=IN:en\"\n\n"
            "Filtering for entities/tournaments happens later using RAG + Gatekeeper\n\n"
            "------------------------------------\n"
            "PREDEFINED TAGS (FLAT LIST)\n"
            "------------------------------------\n"
            "If the list is non-empty, you MUST:\n"
            "- Choose 1-5 tags from this list that fit the alert (field: `tags`).\n"
            "- If no tags fit well, propose new short reusable tags (field: `custom_tags`).\n"
            "- Never output tags outside this provided list into `tags`.\n\n"
            f"predefined_tags = {predefined_tags_json}\n\n"
            "------------------------------------\n"
            "TASK BREAKDOWN (MANDATORY STEPS)\n"
            "------------------------------------\n\n"
            "1️⃣ DEEP INTENT PARSING\n"
            "- Interpret main category, subcategories, followup_questions, custom_question\n"
            "- Identify all domain-specific entities like teams, tournaments, organizations, countries\n"
            "- Resolve pronouns contextually (e.g., \"my team\" → infer best subject entity)\n"
            "- Detect sentiment restrictions (e.g., \"only win\", \"no loss\", \"no controversies\")\n"
            "- If unspecified, infer reasonable defaults using domain knowledge\n\n"
            "2️⃣ EVENT UNDERSTANDING\n"
            "Convert natural language into structured rule-logic:\n"
            "- Allowed event types (live score, match report, squad announcement, etc.)\n"
            "- Specific tournament types (World Cup, Test, ODI, Qualifiers, etc.)\n"
            "- Status constraints: win required? close match? knockout only?\n"
            "- Gender constraints if relevant (men's vs women's)\n"
            "- Geography constraints if implied (country-based loyalty)\n\n"
            "3️⃣ RAG RETRIEVAL PLAN\n"
            "Must:\n"
            "- Expand entity list using synonyms, nicknames, short forms, related terms\n"
            "- Generate weighted keyword clusters:\n"
            "  • primary_keywords = highly important terms\n"
            "  • secondary_keywords = supportive but optional\n"
            "- vector_search_topics must be semantic groups for embeddings search\n"
            "- Set time_filter based on urgency implied in preference:\n"
            "  realtime → matches/events\n"
            "  1d → daily alerts\n"
            "  7d → slower categories\n\n"
            "4️⃣ FILTERING RULES\n"
            "Remove noise aggressively:\n"
            "- Exclude unrelated entities like other sports\n"
            "- Exclude forbidden_topics: loss, defeat, gossip — unless user asked\n"
            "- Avoid contradictions (e.g., don't return \"loss\" if win_required=true)\n\n"
            "5️⃣ TAGGING ALGORITHM (STRICT)\n"
            "- You are given a flat list: predefined_tags.\n"
            "- Build tags by selecting 1-5 items ONLY from predefined_tags that best represent the alert.\n"
            "- Use case-insensitive matching against: category, sub_categories, followup_questions, canonical_entities, contextual_query.\n"
            "- If nothing fits well, leave tags empty and instead fill custom_tags with 1-5 short, reusable labels derived from the same inputs.\n"
            "- Never output any non-predefined item in tags. Put such items into custom_tags.\n"
            "- Never return both tags and custom_tags empty. At least one of them must have 1-5 items.\n\n"
            "6️⃣ PRODUCT/FINANCE INTENT (IF APPLICABLE)\n"
            "- If the alert implies product launch/announcement (e.g., 'launch', 'announces', 'unveils', 'new product'), reflect it via tags or custom_tags.\n"
            "- If the alert implies funding/investment (e.g., 'funding', 'seed', 'Series A/B/C', 'round', 'venture capital'), reflect it via tags or custom_tags.\n\n"
        )

        schema_block = (
            "------------------------------------\n"
            "MUST FOLLOW THIS EXACT JSON SCHEMA\n"
            "------------------------------------\n\n"
            "{\n"
            "  \"alert_id\": \"<same alert_id>\",\n"
            "  \"user_id\": \"<same user_id>\",\n"
            "  \"google_rss_url\": \"https://news.google.com/rss/search?q=<category>&hl=en-IN&gl=IN&ceid=IN:en\",\n"
            "  \"canonical_entities\": [...],\n"
            "  \"expanded_entities\": [...],\n"
            "  \"event_conditions\": [\n"
            "    {\n"
            "      \"entity\": \"<entity_name>\",\n"
            "      \"status\": \"<win|loss|any>\",\n"
            "      \"tournament\": \"<worldcup|test|ODI|any>\"\n"
            "    }\n"
            "  ],\n"
            "  \"allowed_news_types\": [...],\n"
            "  \"forbidden_topics\": [...],\n"
            "  \"retrieval_graph\": {\n"
            "    \"primary_keywords\": [...],\n"
            "    \"secondary_keywords\": [...],\n"
            "    \"vector_search_topics\": [...],\n"
            "    \"time_filter\": \"realtime|1d|7d\"\n"
            "  },\n"
            "  \"strict_matching_rules\": {\n"
            "    \"win_required\": true|false,\n"
            "    \"men_only\": true|false,\n"
            "    \"same_country_opponent_not_required\": true|false\n"
            "  },\n"
            "  \"contextual_query\": \"expanded search query with all synonyms and related terms\",\n"
            "  \"tags\": [\"<predefined_tag>\", \"<predefined_tag>\", ...],\n"
            "  \"custom_tags\": [\"<new_concise_tag>\", ...]\n"
            "}\n\n"
        )

        tail = (
            "------------------------------------\n"
            "RULES — FOLLOW STRICTLY\n"
            "------------------------------------\n"
            "✔ Fully dynamic — never hardcode categories or teams\n"
            "✔ Google RSS URL MUST use only category\n"
            "✔ Everything else filtered by gatekeeper + RAG rules\n"
            "✔ Validate every constraint logically\n"
            "✔ Ensure returned JSON is valid + complete\n"
            "✔ Return ONLY the JSON object, no markdown formatting or explanations\n"
            "✔ No hallucination of teams/tournaments not in input"
        )

        prompt = head + schema_block + tail

        
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON response
        result = json.loads(response_text)

        # Normalize/derive tags using predefined list as ground truth
        try:
            def _flatten_tags_local(obj):
                flat = []
                if isinstance(obj, list):
                    for x in obj:
                        if isinstance(x, str):
                            flat.append(x)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        flat.extend(_flatten_tags_local(v))
                return flat

            predef_list = _flatten_tags_local(predefined_tags or [])
            # Build lookup preserving original case
            lc_to_orig = {t.lower(): t for t in predef_list}

            # Filter LLM-provided tags to predefined only
            llm_tags = [t for t in (result.get("tags") or []) if isinstance(t, str)]
            filtered_llm_tags = []
            for t in llm_tags:
                key = t.strip().lower()
                if key in lc_to_orig and lc_to_orig[key] not in filtered_llm_tags:
                    filtered_llm_tags.append(lc_to_orig[key])

            # If empty, derive from alert text/fields using predefined list
            if not filtered_llm_tags and predef_list:
                hay_parts = []
                hay_parts.extend([str(x) for x in (result.get("canonical_entities") or [])])
                hay_parts.extend([str(x) for x in (result.get("event_conditions") or [])])
                hay_parts.extend([str(x) for x in (result.get("allowed_news_types") or [])])
                hay_parts.extend([str(x) for x in (result.get("forbidden_topics") or [])])
                # Include user inputs
                hay_parts.extend([str(x) for x in (user_preferences.get("sub_categories") or [])])
                hay_parts.extend([str(x) for x in (user_preferences.get("followup_questions") or [])])
                hay_parts.append(user_preferences.get("custom_question") or "")
                hay_parts.append(result.get("contextual_query") or "")
                hay = " \n ".join(hay_parts).lower()

                derived = []
                for tag in predef_list:
                    tl = tag.lower()
                    if tl and tl in hay and tag not in derived:
                        derived.append(tag)
                    if len(derived) >= 5:
                        break
                filtered_llm_tags = derived

            # Build custom_tags from user/LLM signals that are not predefined
            custom = [t for t in (result.get("custom_tags") or []) if isinstance(t, str)]
            seen_custom = set(x.strip().lower() for x in custom)

            def _push_custom(val: str):
                if not val:
                    return
                key = str(val).strip().lower()
                if key and (key not in lc_to_orig) and (val not in filtered_llm_tags) and (key not in seen_custom):
                    custom.append(val)
                    seen_custom.add(key)

            for e in [str(x) for x in (result.get("canonical_entities") or [])]:
                _push_custom(e)
            for s in [str(x) for x in (user_preferences.get("sub_categories") or [])]:
                _push_custom(s)
            for f in [str(x) for x in (user_preferences.get("followup_questions") or [])]:
                _push_custom(f)

            # Ensure at least one of tags/custom_tags has values
            result["tags"] = filtered_llm_tags[:5]
            if not result["tags"] and not custom:
                # As a last resort, put category as a custom tag
                _push_custom(user_preferences.get("category") or "")
            result["custom_tags"] = custom[:5]
        except Exception:
            # If normalization fails, keep whatever LLM returned
            pass
        
        # Derive sport from category/sub_categories
        category_lower = (user_preferences.get("category") or "").lower()
        subs_lower = [s.lower() for s in user_preferences.get("sub_categories", [])]
        sport = None
        mapping = get_trusted_sport_sources()
        for candidate in (category_lower, *subs_lower):
            if candidate in mapping:
                sport = candidate
                break

        # Build trusted queries per entity (site-scoped)
        entities = result.get("canonical_entities") or result.get("expanded_entities") or []
        trusted_queries = []
        trusted_sources = []
        if sport:
            spec = mapping.get(sport)
            if spec:
                trusted_sources.append(spec.get("trusted_source"))
            for ent in entities:
                q = build_trusted_query(sport, ent)
                if q:
                    trusted_queries.append(q)

        # Extract simple custom triggers from custom_question/followups
        def _extract_triggers(text: str) -> list[str]:
            if not text:
                return []
            t = text.lower()
            triggers = []
            # Cricket examples
            if any(phrase in t for phrase in ["last two over", "last 2 over", "death over", "final two overs"]):
                triggers.append("cricket_last_two_overs")
            if any(name in t for name in ["kohli", "virat kohli"]):
                triggers.append("entity_kohli")
            # Football examples
            if any(phrase in t for phrase in ["goal", "red card", "penalty", "injury time"]):
                triggers.append("football_in_match_event")
            return triggers

        custom_triggers = []
        cq = user_preferences.get("custom_question") or ""
        custom_triggers.extend(_extract_triggers(cq))
        for fq in user_preferences.get("followup_questions", []) or []:
            custom_triggers.extend(_extract_triggers(fq))

        # Return the parsed result with all required fields + trusted search helpers
        return {
            "canonical_entities": result.get("canonical_entities", []),
            "expanded_entities": result.get("expanded_entities", []),
            "event_conditions": result.get("event_conditions", []),
            "allowed_news_types": result.get("allowed_news_types", []),
            "forbidden_topics": result.get("forbidden_topics", []),
            "retrieval_graph": result.get("retrieval_graph", {
                "primary_keywords": [],
                "secondary_keywords": [],
                "vector_search_topics": [],
                "time_filter": "1d"
            }),
            "strict_matching_rules": result.get("strict_matching_rules", {
                "win_required": False,
                "men_only": False,
                "same_country_opponent_not_required": True
            }),
            "contextual_query": result.get("contextual_query", ""),
            # Keep backward compatibility
            "category": user_preferences.get("category", ""),
            "sub_categories": user_preferences.get("sub_categories", []),
            "followup_questions": user_preferences.get("followup_questions", []),
            "custom_question": user_preferences.get("custom_question", ""),
            # Tagging outputs
            "tags": [t for t in (result.get("tags") or []) if isinstance(t, str)],
            "custom_tags": [t for t in (result.get("custom_tags") or []) if isinstance(t, str)],
            # Trusted search augmentations
            "trusted_sources": list(dict.fromkeys([s for s in trusted_sources if s])),
            "trusted_queries": trusted_queries,
            # Custom event triggers derived from natural language
            "custom_triggers": list(dict.fromkeys(custom_triggers)),
        }
    except json.JSONDecodeError as je:
        print(f"Error parsing LLM response as JSON: {str(je)}")
        print(f"Response text: {response_text[:500]}")
        return {}
    except Exception as e:
        print(f"Error generating fields: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

async def parse_alerts():
    # Create the alertspars collection if it doesn't exist
    db = client.stagin_local
    alertspars_collection = db.alertspars
    
    # Hardcoded hierarchical predefined tags JSON (provided by user). Supports nested dicts/lists.
    PREDEFINED_TAGS_HIER = {
        "Sports": {
            "Cricket": [
                "T20",
                "ODI",
                "Test",
                "IPL",
                "World Cup",
                "Asia Cup",
                "West Indies",
                "Shai Hope",
                "India National Team",
                "Mumbai Indians",
                "Chennai Super Kings",
                "Royal Challengers Bengaluru",
                "Kolkata Knight Riders",
                "Virat Kohli",
                "Rohit Sharma",
                "Hardik Pandya",
                "MS Dhoni"
            ],
            "Football": [
                "FIFA World Cup",
                "UEFA Champions League",
                "La Liga",
                "Premier League",
                "Indian Super League (ISL)",
                "Argentina",
                "Brazil",
                "Germany",
                "Portugal",
                "Real Madrid",
                "FC Barcelona",
                "Manchester United",
                "Cristiano Ronaldo",
                "Lionel Messi",
                "Neymar Jr"
            ],
            "Basketball": [
                "NBA",
                "EuroLeague",
                "Team USA",
                "Los Angeles Lakers",
                "Golden State Warriors",
                "Chicago Bulls",
                "Lebron James",
                "Stephen Curry",
                "Giannis Antetokounmpo"
            ]
        },
        "Movies & TV": {
            "Bollywood": [
                "Action Movies",
                "Comedy Movies",
                "Romantic Movies",
                "Thriller Movies",
                "Akshay Kumar",
                "Shah Rukh Khan",
                "Salman Khan",
                "Ranveer Singh",
                "Katrina Kaif",
                "Alia Bhatt",
                "OTT Releases",
                "New Trailers"
            ],
            "Hollywood": [
                "Marvel",
                "DC",
                "Horror Movies",
                "Sci-Fi Movies",
                "Oscar Nominees",
                "Netflix Originals",
                "Tom Cruise",
                "Robert Downey Jr.",
                "Dwayne Johnson",
                "Jennifer Lawrence"
            ],
            "Web Series": [
                "Netflix Web Series",
                "Prime Video Web Series",
                "Indian Web Series",
                "Korean Dramas",
                "Crime Series",
                "Comedy Series",
                "Stranger Things",
                "Mirzapur",
                "Sacred Games"
            ]
        },
        "News": {
            "Politics": [
                "Indian Parliament",
                "Government Schemes",
                "Elections",
                "Policies & Bills",
                "PM Modi Updates",
                "Supreme Court Decisions"
            ],
            "Technology": [
                "AI",
                "Smartphones",
                "Startups",
                "Electric Vehicles",
                "Cybersecurity",
                "Space Tech (ISRO)"
            ],
            "Sports News": [
                "Match Scores",
                "Player Injuries",
                "Tournament Updates",
                "Transfer News",
                "Rankings"
            ],
            "Business": [
                "Stock Market",
                "Crypto",
                "Startup Funding",
                "Economy",
                "Brands & CEOs"
            ]
        },
        "YouTube": {
            "Tech": [
                "Unboxing Videos",
                "Tech Reviews",
                "Gadget Comparisons",
                "Laptop & Mobile Launches",
                "How-to Tutorials"
            ],
            "Gaming": [
                "BGMI",
                "Valorant",
                "PUBG Mobile",
                "eSports Tournaments",
                "Gameplay Highlights",
                "Live Streams"
            ],
            "Travel": [
                "Indian Travel",
                "International Travel",
                "Vlogs",
                "Budget Travel Tips",
                "Adventure Travel"
            ],
            "Education": [
                "UPSC Prep",
                "GK & Current Affairs",
                "NEET/JEE",
                "Coding Tutorials",
                "Motivation"
            ]
        },
        "Custom": {
            "User Interests": []
        }
    }

    def _flatten_tags(obj):
        flat = []
        try:
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, str):
                        flat.append(x)
            elif isinstance(obj, dict):
                for v in obj.values():
                    flat.extend(_flatten_tags(v))
        except Exception:
            pass
        # Deduplicate while preserving order
        seen = set()
        out = []
        for t in flat:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    flat_tags = _flatten_tags(PREDEFINED_TAGS_HIER)
    
    # Clear existing parsed alerts if needed
    # await alertspars_collection.delete_many({})
    
    # Process each alert in the alerts collection
    async for alert in alerts_collection.find():
        try:
            # Generate fields using LLM
            alert_content = json.dumps(alert, default=str)
            generated_fields = await generate_alert_fields(alert_content, predefined_tags=flat_tags)
            
            # Create the parsed alert structure
            parsed_alert = {
                "alert_id": str(alert["_id"]),
                **generated_fields,
                "original_alert": alert  # Keep the original alert data
            }
            
            # Insert the parsed alert into the new collection
            await alertspars_collection.update_one(
                {"alert_id": str(alert["_id"])},
                {"$set": parsed_alert},
                upsert=True
            )
            print(f"Processed alert: {alert['_id']}")
            
        except Exception as e:
            print(f"Error processing alert {alert.get('_id', 'unknown')}: {str(e)}")

async def main():
    try:
        await parse_alerts()
        print("Alert parsing completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        try:
            import inspect
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                res = close_fn()
                if inspect.iscoroutine(res):
                    await res
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
