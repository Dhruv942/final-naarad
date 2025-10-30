import pytest

import controllers.rag_news_controller as controller


class _AsyncCursor:
    def __init__(self, items):
        self._items = items

    async def to_list(self, _):
        return self._items


class _AsyncCollection:
    async def count_documents(self, query):
        return 0

    async def update_one(self, *args, **kwargs):
        return None

    async def find_one(self, *args, **kwargs):
        return None


class _DBMock:
    def __init__(self):
        self._collections = {}

    def get_collection(self, name):
        if name not in self._collections:
            self._collections[name] = _AsyncCollection()
        return self._collections[name]


@pytest.mark.asyncio
async def test_description_comes_from_gatekeeper_and_is_cleaned(monkeypatch):
    fake_alert = {
        "_id": "690086856d5f96804e6d17f9",
        "user_id": "7ef93f95-626d-4488-8732-7f39edd282e6",
        "is_active": True,
        "main_category": "News",
        "sub_categories": ["stockmarket"],
        "followup_questions": [],
        "custom_question": "",
    }

    class _AlertsCollection:
        async def count_documents(self, query):
            return 1

        def find(self, query):
            return _AsyncCursor([fake_alert])

    monkeypatch.setattr(controller, "alerts_collection", _AlertsCollection())
    monkeypatch.setattr(controller, "db", _DBMock())

    llm_title = "USD/INR: ₹88.614"
    llm_desc_dirty = (
        "2 hours ago ... The current USD/INR exchange rate is 88.614. "
        "What Is the Daily Range for USD/INR? Today's ..."
    )

    pipeline_result = {
        "status": "success",
        "articles": [
            {
                "title": llm_title,
                "content": "Some content that might include site UI.",
                "url": "https://in.investing.com/currencies/usd-inr",
                "source": "in.investing.com",
                "published_at": None,
                "metadata": {
                    "generated_title": llm_title,
                    "generated_description": llm_desc_dirty,
                    "relevance_score": 0.9,
                    "topic": "general",
                },
            }
        ],
        "gatekeeper_stats": {"approved": 1},
    }

    async def _fake_process_alert(alert_dict):
        return pipeline_result

    monkeypatch.setattr(controller.rag_pipeline, "process_alert", _fake_process_alert)

    result = await controller.get_user_news("7ef93f95-626d-4488-8732-7f39edd282e6")

    assert result["status"] == "success"
    assert result["alerts_processed"] == 1
    assert len(result["results"]) == 1

    item = result["results"][0]
    assert item["title"].startswith("USD/INR")

    desc = item["description"]
    assert desc
    assert "hours ago" not in desc.lower()
    assert "what is the daily range" not in desc.lower()
    assert "..." not in desc

