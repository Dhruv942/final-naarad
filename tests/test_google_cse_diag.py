import os
import sys
import asyncio
import httpx

API = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")
CX = os.getenv("GOOGLE_CX") or os.getenv("GOOGLE_SEARCH_CX")

async def call_cse(query: str, search_type: str | None):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API,
        "cx": CX,
        "q": query[:100],
        "num": 1,
        "safe": "off",
    }
    if search_type:
        params["searchType"] = search_type
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(url, params=params)
            status = r.status_code
            try:
                data = r.json()
            except Exception:
                data = {"_non_json": r.text[:200]}
            items = (data or {}).get("items", [])
            first_link = items[0].get("link") if items else None
            error = (data or {}).get("error")
            print({
                "type": search_type or "web",
                "status": status,
                "has_items": bool(items),
                "first_link": first_link,
                "error": error,
            })
        except Exception as e:
            print({"type": search_type or "web", "exception": f"{type(e).__name__}: {e}"})

async def main():
    q = os.environ.get("TEST_CSE_QUERY") or (" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "USD/INR today investing")
    print({"api_present": bool(API), "cx_present": bool(CX), "query": q})
    await call_cse(q, None)
    await call_cse(q, "image")

if __name__ == "__main__":
    asyncio.run(main())
