import asyncio
import os
from typing import List, Dict, Any

from services.news_fetcher_service import NewsFetcherService


async def run_test(query: str) -> None:
    fetcher = NewsFetcherService()
    print(f"Testing SERP news fallback for query: {query}")
    try:
        results: List[Dict[str, Any]] = await fetcher._fetch_serp_fallback(query)
    except Exception as e:
        print(f"ERROR: SERP fallback raised: {type(e).__name__}: {e}")
        return

    print(f"Total results: {len(results)}")
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r.get('title','')[:90]}\n   {r.get('url','')}")


if __name__ == "__main__":
    q = os.environ.get("TEST_SERP_QUERY", "USD/INR rate today site:in.investing.com OR site:reuters.com")
    asyncio.run(run_test(q))
