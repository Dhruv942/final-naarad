import asyncio
import os

from services.news_fetcher_service import NewsFetcherService


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return value[:3] + "***" + value[-3:]


async def run_test(query: str) -> None:
    fetcher = NewsFetcherService()
    api = fetcher.google_api_key
    cx = fetcher.google_cx

    env_api = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")
    env_cx = os.getenv("GOOGLE_CX") or os.getenv("GOOGLE_SEARCH_CX")

    source_api = "env" if env_api else ("config" if api else "missing")
    source_cx = "env" if env_cx else ("config" if cx else "missing")

    print(f"GOOGLE_API_KEY: source={source_api}, value={_mask(api or '')}")
    print(f"GOOGLE_CX: source={source_cx}, value={_mask(cx or '')}")

    if not api or not cx:
        print("SKIP: GOOGLE_API_KEY/GOOGLE_CX not configured. Set in config.py or env and retry.")
        return

    print(f"Testing Google Images search for: {query}")
    url = await fetcher._search_google_image(query)
    if url:
        print(f"OK: Found image URL: {url}")
    else:
        print("FAIL: No image URL returned (check quota/keys/network)")


if __name__ == "__main__":
    q = os.environ.get("TEST_IMAGE_QUERY", "USD/INR exchange rate today")
    asyncio.run(run_test(q))
