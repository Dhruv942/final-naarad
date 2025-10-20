"""
Enhanced Gemini LLM Judge with Chain-of-Thought Reasoning
Provides intelligent, explainable article selection with deep reasoning
"""
import logging
from typing import List, Dict, Any
import json

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

from .config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class GeminiJudge:
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        if genai is None:
            raise RuntimeError("google-generativeai not installed")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def judge(self, alert: Dict, candidates: List[Dict], max_select: int = 3) -> List[Dict]:
        """
        Use Gemini with chain-of-thought reasoning to understand user intent 
        and select the best matching articles with detailed rationales.
        """
        try:
            category = (alert or {}).get("main_category", "")
            keywords = (alert or {}).get("sub_categories", []) or []
            followups = (alert or {}).get("followup_questions", []) or []
            custom = (alert or {}).get("custom_question", "") or ""

            # Build enhanced chain-of-thought prompt
            prompt = self._build_cot_prompt(category, keywords, followups, custom, candidates, max_select)
            
            # Generate with higher temperature for more creative reasoning
            generation_config = genai.types.GenerationConfig(
                temperature=0.4,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            text = response.text.strip()

            # Parse JSON output with chain-of-thought
            try:
                # Extract JSON from response (might have markdown formatting)
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_text = text[json_start:json_end]
                    result = json.loads(json_text)
                else:
                    result = json.loads(text)
                
                selected_idxs = result.get("selected", [])
                rationales = result.get("rationales", {})
                reasoning = result.get("reasoning", "")
                diversity_score = result.get("diversity_score", 0.5)
            except Exception as e:
                logger.warning(f"JSON parsing failed: {e}, using fallback")
                # fallback: return first N
                selected_idxs = list(range(min(max_select, len(candidates))))
                rationales = {}
                reasoning = "Fallback selection"
                diversity_score = 0.5

            out = []
            for idx in selected_idxs:
                if 0 <= idx < len(candidates):
                    it = dict(candidates[idx])
                    it["judge_rationale"] = rationales.get(str(idx), "Selected by Gemini")
                    it["judge_reasoning"] = reasoning
                    it["diversity_score"] = diversity_score
                    out.append(it)
            
            logger.info(f"🤖 Gemini judge reasoning: {reasoning[:100]}...")
            return out[:max_select]
        except Exception as e:
            logger.warning(f"Gemini judge failed: {e}")
            return candidates[:max_select]

    def _build_prompt(self, category: str, keywords: List[str], followups: List[str], custom: str, candidates: List[Dict], max_select: int) -> str:
        intent_parts = []
        if category:
            intent_parts.append(f"Category: {category}")
        if keywords:
            intent_parts.append(f"Keywords: {', '.join(keywords)}")
        if followups:
            intent_parts.append(f"Follow-up interests: {', '.join(followups)}")
        if custom:
            intent_parts.append(f"Custom request: {custom}")
        intent_text = "\n".join(intent_parts) if intent_parts else "General news"

        cand_json = []
        for i, c in enumerate(candidates):
            cand_json.append({
                "idx": i,
                "title": c.get("title", ""),
                "summary": c.get("summary", ""),
                "published_date": c.get("published_date", ""),
            })

        prompt = f"""You are an intelligent news curator. Your job is to understand the user's intent and select the most relevant articles.

**User Intent:**
{intent_text}

**Candidate Articles:**
{json.dumps(cand_json, indent=2)}

**Task:**
1. Understand the user's intent deeply (consider keywords, follow-ups, and custom requests).
2. Select up to {max_select} articles that best match the intent.
3. Provide a brief rationale for each selected article explaining why it matches the user's needs.

**Output Format (JSON only, no extra text):**
{{
  "selected": [<list of article indices>],
  "rationales": {{
    "<idx>": "<brief rationale for why this article matches user intent>"
  }}
}}
"""
        return prompt

    def _build_cot_prompt(self, category: str, keywords: List[str], followups: List[str], 
                          custom: str, candidates: List[Dict], max_select: int) -> str:
        """Build chain-of-thought prompt for enhanced reasoning"""
        intent_parts = []
        if category:
            intent_parts.append(f"Category: {category}")
        if keywords:
            intent_parts.append(f"Keywords: {', '.join(keywords)}")
        if followups:
            intent_parts.append(f"Follow-up interests: {', '.join(followups)}")
        if custom:
            intent_parts.append(f"Custom request: {custom}")
        intent_text = "\n".join(intent_parts) if intent_parts else "General news"

        cand_json = []
        for i, c in enumerate(candidates):
            article_data = {
                "idx": i,
                "title": c.get("title", ""),
                "summary": c.get("summary", "")[:200],  # Limit summary length
                "published_date": c.get("published_date", ""),
            }
            
            # Add NLP intelligence if available
            if "nlp_intelligence" in c:
                nlp = c["nlp_intelligence"]
                article_data["sentiment"] = nlp.get("sentiment", {}).get("label", "NEUTRAL")
                article_data["quality_score"] = round(nlp.get("quality_score", 0.5), 2)
                article_data["key_topics"] = nlp.get("key_topics", [])[:3]
            
            # Add personalization score if available
            if "personalization_score" in c:
                article_data["personalization"] = round(c["personalization_score"], 2)
            
            cand_json.append(article_data)

        prompt = f"""You are an advanced AI news curator with deep understanding of user intent and content quality.
Your goal is to select the most relevant and diverse articles using chain-of-thought reasoning.

**User Intent:**
{intent_text}

**Candidate Articles:**
{json.dumps(cand_json, indent=2)}

**Task - Think Step by Step:**

1. **Intent Analysis**: What is the user truly looking for? Consider explicit keywords and implicit needs.

2. **Article Evaluation**: For each article, assess:
   - Relevance to user intent (keywords, topics, custom requests)
   - Content quality (sentiment, writing quality, informativeness)
   - Freshness and timeliness
   - Uniqueness and diversity from other candidates

3. **Selection Strategy**: Choose up to {max_select} articles that:
   - Best match the user's intent
   - Provide diverse perspectives or information
   - Are high quality and engaging
   - Balance relevance with variety

4. **Rationale**: For each selected article, explain WHY it was chosen and HOW it serves the user's needs.

**Output Format (JSON only, no markdown):**
{{
  "reasoning": "<Your chain-of-thought analysis: intent understanding + selection strategy>",
  "selected": [<list of article indices, e.g., [0, 3, 5]>],
  "rationales": {{
    "<idx>": "<specific rationale for why this article matches user intent>"
  }},
  "diversity_score": <0-1 score indicating how diverse your selection is>
}}

Think carefully and provide your response:"""
        return prompt


def get_gemini_judge() -> GeminiJudge | None:
    try:
        if genai is not None and GEMINI_API_KEY:
            return GeminiJudge()
    except Exception:
        pass
    return None
