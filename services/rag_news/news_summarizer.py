"""
News Summarizer Module

Takes fetched news articles and generates concise, actionable summaries
that directly answer user's questions.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
from .config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class NewsSummarizer:
    """Generates concise summaries from news articles"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    async def generate_summary(
        self,
        articles: List[Dict[str, Any]],
        alert_data: Dict,
        entity: str
    ) -> str:
        """
        Generate a concise summary answering user's specific question

        Args:
            articles: List of news articles
            alert_data: User's alert preferences
            entity: Specific entity (e.g., "US Dollar", "Yes Bank")

        Returns:
            Concise summary with specific information
        """
        try:
            # Extract key information from articles
            article_texts = []
            for idx, article in enumerate(articles[:5], 1):  # Top 5 articles
                text = f"{idx}. {article.get('title', '')} - {article.get('snippet', '')}"
                article_texts.append(text)

            combined_articles = "\n\n".join(article_texts)

            # Get user's question
            custom_question = alert_data.get('custom_question', '')
            contextual_query = alert_data.get('contextual_query', '')

            # Create summarization prompt
            prompt = f"""You are a financial news summarizer. Create a concise, actionable summary from the articles below.

USER'S QUESTION: {custom_question}
ENTITY: {entity}
CONTEXT: {contextual_query}

ARTICLES:
{combined_articles}

INSTRUCTIONS:
1. If the entity is "US Dollar" and user asked for daily price:
   - Provide the EXACT CURRENT PRICE in INR (e.g., "1 USD = 84.15 INR today")
   - Mention if it increased or decreased from yesterday
   - Keep it ONE LINE

2. If the entity is "Yes Bank" and user asked for stock price:
   - Provide the EXACT CURRENT STOCK PRICE (e.g., "Yes Bank: ₹22.50")
   - Mention % change and analyst view in ONE LINE

3. If current price is NOT available in articles:
   - Say "Current {entity} price not available in latest news"
   - Provide the most relevant insight from articles in ONE LINE

4. Format:
   - MAX 2 sentences
   - Start with the price/number if available
   - No fluff, direct answer only

OUTPUT FORMAT:
Title: [One line title with price]
Summary: [One line actionable summary]

Generate the summary now:"""

            # Call LLM
            response = self.model.generate_content(prompt)
            summary_text = response.text.strip()

            # Parse response
            lines = summary_text.split('\n')
            title = ""
            summary = ""

            for line in lines:
                if line.startswith('Title:'):
                    title = line.replace('Title:', '').strip()
                elif line.startswith('Summary:'):
                    summary = line.replace('Summary:', '').strip()

            if not title or not summary:
                # Fallback parsing
                if len(lines) >= 2:
                    title = lines[0].replace('Title:', '').strip()
                    summary = lines[1].replace('Summary:', '').strip()
                else:
                    title = f"{entity} Update"
                    summary = summary_text[:200]

            return {
                'title': title,
                'summary': summary,
                'entity': entity,
                'source_count': len(articles),
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'title': f"{entity} Update",
                'summary': f"Unable to generate summary for {entity}",
                'entity': entity,
                'source_count': 0,
                'generated_at': datetime.utcnow().isoformat()
            }


# Singleton instance
_summarizer = None

def get_news_summarizer() -> NewsSummarizer:
    """Get or create NewsSummarizer instance"""
    global _summarizer
    if _summarizer is None:
        _summarizer = NewsSummarizer()
    return _summarizer
