import asyncio
import os
import sys

from services.news_fetcher_service import NewsFetcherService


async def main() -> None:
    query = os.environ.get("TEST_IMAGE_QUERY") or (" ".join(sys.argv[1:]) if len(sys.argv) > 1 else None)
    if not query:
        print("")
        return
    fetcher = NewsFetcherService()
    url = await fetcher._search_google_image(query)
    print(url or "")


if __name__ == "__main__":
    asyncio.run(main())
