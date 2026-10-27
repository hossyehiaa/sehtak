"""
Medical Web Search Agent for Sehtak AI (`app/ai/tools/web_search_agent.py`).
Enables real-time retrieval of active ingredients, clinical guidelines, dosages,
and drug safety monographs from live web authorities (EDA, WHO, NIH, Drugs.com, etc.).
"""

import html
import logging
import re
from typing import Dict, List, Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class MedicalWebSearchAgent:
    """Autonomous agent capable of live medical and pharmacological web search."""

    def __init__(self):
        self.enabled = settings.ENABLE_LIVE_WEB_SEARCH
        self.tavily_api_key = settings.TAVILY_API_KEY

    async def search(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """
        Executes a targeted medical web search.
        Tier 1: Tavily API (if key is set).
        Tier 2: DuckDuckGo HTML parser (zero-config, high resilience).
        """
        if not self.enabled or not query or len(query.strip()) < 2:
            return []

        cleaned_query = query.strip()
        # 1. Try Tavily if configured
        if self.tavily_api_key:
            try:
                results = await self._search_tavily(cleaned_query, max_results)
                if results:
                    return results
            except Exception as e:
                logger.warning("Tavily search failed: %s. Falling back to DuckDuckGo.", e)

        # 2. DuckDuckGo Search
        try:
            return await self._search_duckduckgo(cleaned_query, max_results)
        except Exception as e:
            logger.warning("DuckDuckGo medical search failed: %s", e)
            return []

    async def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, str]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": f"{query} medication drug pharmacology indication dosage",
            "search_depth": "advanced",
            "max_results": max_results,
            "include_domains": [
                "nih.gov", "who.int", "drugs.com", "medscape.com",
                "edaegypt.gov.eg", "mayoclinic.org", "webmd.com", "ncbi.nlm.nih.gov"
            ],
        }
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", "Medical Reference"),
                        "snippet": item.get("content", "").strip(),
                        "url": item.get("url", ""),
                        "source": self._extract_domain(item.get("url", "")),
                    })
                return results
        return []

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Scrapes DuckDuckGo HTML for authoritative medical facts."""
        search_terms = f"{query} دواعي الاستعمال المادة الفعالة الجرعة"
        headers = {"User-Agent": USER_AGENT}
        data = {"q": search_terms}

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.post("https://html.duckduckgo.com/html/", data=data, headers=headers)
            if resp.status_code != 200:
                # Retry with English pharmacology keywords
                data = {"q": f"{query} active ingredient medical indication pharmacology"}
                resp = await client.post("https://html.duckduckgo.com/html/", data=data, headers=headers)
                if resp.status_code != 200:
                    return []

            html_text = resp.text

            # Extract snippets and titles
            # Pattern matches DDG HTML result snippet: <a class="result__snippet..." href="...">...</a>
            snippet_matches = re.findall(
                r"<a[^>]*class=\"result__snippet[^\"]*\"[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a>",
                html_text,
                re.DOTALL,
            )

            # If href is not inside snippet, extract separately
            if not snippet_matches:
                raw_snippets = re.findall(
                    r"<a[^>]*class=\"result__snippet[^\"]*\"[^>]*>(.*?)</a>",
                    html_text,
                    re.DOTALL,
                )
                raw_links = re.findall(
                    r"<a[^>]*class=\"result__url[^\"]*\"[^>]*href=\"([^\"]*)\"",
                    html_text,
                    re.DOTALL,
                )
                results = []
                for idx, snip in enumerate(raw_snippets[:max_results]):
                    clean_text = self._clean_html(snip)
                    link = raw_links[idx] if idx < len(raw_links) else ""
                    if len(clean_text) > 20:
                        results.append({
                            "title": f"Medical Source #{idx+1}",
                            "snippet": clean_text,
                            "url": link,
                            "source": self._extract_domain(link),
                        })
                return results

            results = []
            for link, snip in snippet_matches[:max_results]:
                clean_text = self._clean_html(snip)
                if len(clean_text) > 20:
                    results.append({
                        "title": f"Medical Source",
                        "snippet": clean_text,
                        "url": link,
                        "source": self._extract_domain(link),
                    })
            return results

    def _clean_html(self, raw_html: str) -> str:
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        clean = html.unescape(clean)
        return re.sub(r"\s+", " ", clean).strip()

    def _extract_domain(self, url: str) -> str:
        if not url:
            return "Medical Source"
        # Match domain name
        m = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if m:
            return m.group(1)
        return "Web Reference"

    def format_for_prompt(self, results: List[Dict[str, str]]) -> str:
        """Formats search results as an evidence block for the LLM."""
        if not results:
            return ""
        lines = ["--- LIVE MEDICAL WEB SEARCH EVIDENCE (REAL-TIME AGENT RETRIEVAL) ---"]
        for idx, r in enumerate(results, 1):
            lines.append(f"[{idx}] Source: {r['source']} ({r['url']})")
            lines.append(f"    Snippet: {r['snippet']}")
        lines.append("---------------------------------------------------------------------")
        return "\n".join(lines)


web_search_agent = MedicalWebSearchAgent()
