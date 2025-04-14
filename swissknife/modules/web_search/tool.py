from swissknife.modules.web_search.service import TavilySearchService


def get_web_search_tool_definition(provider="claude"):
    """Return the tool definition for web search based on provider."""
    tool_description = "Searches the web for up-to-date information on a specific topic or query. Use this to gather current information, verify facts, or explore the latest developments. Summarize your findings *before* presenting them to the user. If the user is seeking information that is likely to have changed recently, this is the tool to use."
    tool_arguments = {
        "query": {
            "type": "string",
            "description": "The search query to use for web search. Use precise and specific keywords to get the most relevant results. Include any relevant context to improve search accuracy.",
        },
        "search_depth": {
            "type": "string",
            "enum": ["basic", "advanced"],
            "description": "The depth of the search to perform. 'basic' is faster and suitable for general information. 'advanced' is more thorough and suitable for complex or nuanced queries. Start with 'basic' and only use 'advanced' if initial results are insufficient.",
            "default": "basic",
        },
        "max_results": {
            "type": "integer",
            "description": "The maximum number of search results to return. A higher number allows for broader coverage but may include less relevant results. (Range: 1-10)",
            "default": 10,
            "minimum": 1,
            "maximum": 20,
        },
    }
    tool_required = ["query"]
    if provider == "claude":
        return {
            "name": "web_search",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_web_extract_tool_definition(provider="claude"):
    """Return the tool definition for web content extraction based on provider."""
    tool_description = "Retrieves the content from a web page specified by its URL. Use this to access information, documentation, or data available on the web *after* finding the URL using `web_search` or other means. Only use HTTP/HTTPS URLs. DO NOT use this tool to access local project files. Summarize the content of the webpage before presenting it to the user. Prioritize extracting information relevant to the user's current request."
    tool_arguments = {
        "url": {
            "type": "string",
            "description": "The complete HTTP or HTTPS web address to retrieve content from (e.g., 'https://example.com/page'). Ensure the URL is valid and accessible. Verify the URL's relevance to the user's request before fetching.",
        }
    }
    tool_required = ["url"]
    if provider == "claude":
        return {
            "name": "fetch_webpage",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "fetch_webpage",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_web_search_tool_handler(tavily_service: TavilySearchService):
    """
    Return a handler function for the web search tool.

    Args:
        tavily_service: An instance of TavilySearchService

    Returns:
        Function that handles web search tool calls
    """

    def web_search_handler(**params):
        query = params.get("query")
        search_depth = params.get("search_depth", "basic")
        max_results = params.get("max_results", 10)

        if not query:
            return "Error: No search query provided."

        print(f"🔍 Searching the web for: {query}")
        results = tavily_service.search(
            query=query, search_depth=search_depth, max_results=max_results
        )

        return tavily_service.format_search_results(results)

    return web_search_handler


def get_web_extract_tool_handler(tavily_service: TavilySearchService):
    """
    Return a handler function for the web extract tool.

    Args:
        tavily_service: An instance of TavilySearchService

    Returns:
        Function that handles web extract tool calls
    """

    def web_extract_handler(**params):
        url = params.get("url")

        if not url:
            return "Error: No URL provided."

        print(f"📄 Extracting content from URL: {url}")
        results = tavily_service.extract(url=url)
        return tavily_service.format_extract_results(results)

    return web_extract_handler


def register(service_instance=None, agent=None):
    """
    Register this tool with the central registry or directly with an agent

    Args:
        service_instance: The web search service instance
        agent: Agent instance to register with directly (optional)
    """
    from swissknife.modules.tools.registration import register_tool

    register_tool(
        get_web_search_tool_definition,
        get_web_search_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_web_extract_tool_definition,
        get_web_extract_tool_handler,
        service_instance,
        agent,
    )
