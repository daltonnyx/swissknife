import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from tavily import TavilyClient


class TavilySearchService:
    """Service for interacting with the Tavily Search API using the official SDK."""

    def __init__(self):
        """Initialize the Tavily search service with API key from environment."""
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        self.client = TavilyClient(api_key=self.api_key)

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        include_domains: List[str] | None = None,
        exclude_domains: List[str] | None = None,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.

        Args:
            query: The search query
            search_depth: 'basic' or 'advanced' search depth
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            max_results: Maximum number of results to return

        Returns:
            Dict containing search results
        """
        try:
            params = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
            }

            if include_domains:
                params["include_domains"] = include_domains

            if exclude_domains:
                params["exclude_domains"] = exclude_domains

            return self.client.search(**params)
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return {"error": str(e)}

    def extract(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a specific URL using Tavily API.

        Args:
            url: The URL to extract content from

        Returns:
            Dict containing the extracted content
        """
        try:
            return self.client.extract(url)
        except Exception as e:
            print(f"❌ Extract error: {str(e)}")
            return {"error": str(e)}

    def format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results into a readable string."""
        if "error" in results:
            return f"Search error: {results['error']}"

        formatted_text = f"**Search Results**:  \n\n"

        if "results" in results:
            for i, result in enumerate(results["results"], 1):
                formatted_text += f"{i}. {result.get('title', 'No title')}  \n"
                formatted_text += f"   URL: {result.get('url', 'No URL')}  \n"
                formatted_text += f"   {result.get('content', 'No content')}  \n\n"
        else:
            formatted_text += "No results found."

        return formatted_text

    def format_extract_results(self, results: Dict[str, Any]) -> str:
        """Format extract results into a readable string."""

        if "failed_results" in results and results["failed_results"]:
            result = results["failed_results"][0]
            return f"Extract failed: {result.get('error', 'Unknown error')}"

        if "results" in results and results["results"]:
            result = results["results"][0]
            url = result.get("url", "Unknown URL")
            content = result.get("raw_content", "No content available")
            return f"Extracted content from {url}:\n\n{content}"
        else:
            return "No content could be extracted."
