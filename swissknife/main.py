import click
import importlib
import os
import sys
import traceback
from datetime import datetime
from swissknife.modules.chat import ConsoleUI
from swissknife.modules.gui import ChatWindow
from swissknife.modules.chat import MessageHandler
from swissknife.modules.scraping import ScrapingService
from swissknife.modules.web_search import TavilySearchService
from swissknife.modules.clipboard import ClipboardService
from swissknife.modules.memory import (
    MemoryService,
    ContextPersistenceService,
)
from swissknife.modules.code_analysis import CodeAnalysisService
from swissknife.modules.anthropic import AnthropicService
from swissknife.modules.llm.service_manager import ServiceManager
from swissknife.modules.llm.model_registry import ModelRegistry
from swissknife.modules.coder import SpecPromptValidationService
from swissknife.modules.agents import AgentManager, LocalAgent
from PySide6.QtWidgets import QApplication


@click.group()
def cli():
    """SwissKnife - AI Assistant and Agent Framework"""
    pass


def cli_prod():
    os.environ["MEMORYDB_PATH"] = os.path.expanduser("~/.swissknife/memorydb")
    os.environ["MCP_CONFIG_PATH"] = os.path.expanduser("~/.swissknife/mcp_server.json")
    os.environ["SW_AGENTS_CONFIG"] = os.path.expanduser("~/.swissknife/agents.toml")
    os.environ["PERSISTENCE_DIR"] = os.path.expanduser("~/.swissknife/persistents")
    cli()  # Delegate to main CLI function


@cli.command()
@click.argument("url")
@click.argument("output_file")
@click.option("--summarize", is_flag=True, help="Summarize the content using Claude")
@click.option(
    "--explain", is_flag=True, help="Explain the content for non-experts using Claude"
)
def get_url(url: str, output_file: str, summarize: bool, explain: bool):
    """Fetch URL content and save as markdown"""
    if summarize and explain:
        raise click.UsageError(
            "Cannot use both --summarize and --explain options together"
        )

    try:
        click.echo(f"\n🌐 Fetching content from: {url}")
        scraper = ScrapingService()
        content = scraper.scrape_url(url)
        click.echo("✅ Content successfully scraped")

        if summarize or explain:
            # Create the LLM service
            llm_service = AnthropicService()

            if summarize:
                click.echo("\n🤖 Summarizing content using LLM...")
                content = llm_service.summarize_content(content)
                click.echo("✅ Content successfully summarized")
            else:  # explain
                click.echo("\n🤖 Explaining content using LLM...")
                content = llm_service.explain_content(content)
                click.echo("✅ Content successfully explained")

        click.echo(f"\n💾 Saving content to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        operation = "explained" if explain else "summarized" if summarize else ""
        click.echo(
            f"✅ Successfully saved {operation + ' ' if operation else ''}markdown to {output_file}"
        )
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)


def setup_services(provider):
    # Initialize the model registry and service manager
    registry = ModelRegistry.get_instance()
    manager = ServiceManager.get_instance()

    # Set the current model based on provider
    models = registry.get_models_by_provider(provider)
    if models:
        # Find default model for this provider
        default_model = next((m for m in models if m.default), models[0])
        registry.set_current_model(default_model.id)

    # Get the LLM service from the manager
    llm_service = manager.get_service(provider)

    # Initialize services
    memory_service = MemoryService()
    context_service = ContextPersistenceService()
    clipboard_service = ClipboardService()
    aider_service = SpecPromptValidationService("groq")
    # Try to create search service if API key is available
    try:
        search_service = TavilySearchService()
    except Exception as e:
        click.echo(f"⚠️ Web search tools not available: {str(e)}")
        search_service = None

    # Initialize code analysis service
    try:
        code_analysis_service = CodeAnalysisService()
    except Exception as e:
        click.echo(f"⚠️ Code analysis tool not available: {str(e)}")
        code_analysis_service = None

    # try:
    #     scraping_service = ScrapingService()
    # except Exception as e:
    #     click.echo(f"⚠️ Scraping service not available: {str(e)}")
    #     scraping_service = None

    # Clean up old memories (older than 1 month)
    try:
        removed_count = memory_service.cleanup_old_memories(months=1)
        if removed_count > 0:
            click.echo(f"🧹 Cleaned up {removed_count} old conversation memories")
    except Exception as e:
        click.echo(f"⚠️ Memory cleanup failed: {str(e)}")

    # Register all tools with their respective services
    services = {
        "llm": llm_service,
        "memory": memory_service,
        "clipboard": clipboard_service,
        "code_analysis": code_analysis_service,
        "web_search": search_service,
        "aider": aider_service,
        "context_persistent": context_service,
        # "scraping": scraping_service,
    }
    return services


def setup_agents(services, config_path):
    """
    Set up the agent system with specialized agents.

    Args:
        services: Dictionary of services
    """
    # Get the singleton instance of agent manager
    agent_manager = AgentManager.get_instance()

    # Add agent_manager to services for tool registration
    services["agent_manager"] = agent_manager

    # Get the LLM service
    llm_service = services["llm"]

    # Create specialized agents
    if config_path:
        click.echo("Using command-line argument for agent configuration path.")
        os.environ["SW_AGENTS_CONFIG"] = config_path
    else:
        config_path = os.getenv("SW_AGENTS_CONFIG")
        if not config_path:
            config_path = "./agents.toml"
            # If config path doesn't exist, create a default one
        if not os.path.exists(config_path):
            click.echo(
                f"Agent configuration not found at {config_path}. Creating default configuration."
            )
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            default_config = """# Default SwissKnife Agent Configuration
[[agents]]
name = "default"
description = "Default assistant agent"
system_prompt = \"\"\"You are a helpful AI assistant. Always provide accurate, helpful, and ethical responses.
Current date: {current_date}
\"\"\"
tools = ["memory", "clipboard", "web_search", "code_analysis"]
"""

            with open(config_path, "w+") as f:
                f.write(default_config)

            click.echo(f"Created default agent configuration at {config_path}")
    # Load agents from configuration
    agent_definitions = AgentManager.load_agents_from_config(config_path)
    agent_name = None
    for agent_def in agent_definitions:
        llm_service = services["llm"]
        if not agent_name:
            agent_name = agent_def["name"]
        agent = LocalAgent(
            name=agent_def["name"],
            description=agent_def["description"],
            llm_service=llm_service,
            services=services,
            tools=agent_def["tools"],
        )
        agent.set_system_prompt(
            agent_def["system_prompt"].replace(
                "{current_date}", datetime.today().strftime("%Y-%m-%d")
            )
        )
        agent_manager.register_agent(agent)

    from swissknife.modules.mcpclient.tool import register as mcp_register

    mcp_register()

    # Select the initial agent if specified
    if agent_name:
        if not agent_manager.select_agent(agent_name):
            available_agents = ", ".join(agent_manager.agents.keys())
            click.echo(
                f"⚠️ Unknown agent: {agent_name}. Using default agent. Available agents: {available_agents}"
            )


def discover_and_register_tools(services=None):
    """
    Discover and register all tools

    DEPRECATED: This function is deprecated and will be removed in a future version.
    Use setup_agents() and register_agent_tools() instead.

    Args:
        services: Dictionary mapping service names to service instances
    """
    import warnings

    warnings.warn(
        "discover_and_register_tools is deprecated. Use setup_agents() and register_agent_tools() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if services is None:
        services = {}

    # List of tool modules and their corresponding service keys
    tool_modules = [
        ("modules.memory.tool", "memory"),
        ("modules.code_analysis.tool", "code_analysis"),
        ("modules.clipboard.tool", "clipboard"),
        ("modules.web_search.tool", "web_search"),
        ("modules.coder.tool", "aider"),
    ]

    for module_name, service_key in tool_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "register"):
                service_instance = services.get(service_key)
                module.register(service_instance)
                # print(f"✅ Registered tools from {module_name}")
        except ImportError as e:
            print(f"⚠️ Error importing tool module {module_name}: {e}")

    from swissknife.modules.mcpclient.tool import register as mcp_register

    mcp_register()


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["claude", "groq", "openai", "google", "deepinfra"]),
    default=None,
    help="LLM provider to use (claude, groq, openai, google, or deepinfra)",
)
@click.option(
    "--agent-config", default=None, help="Path to the agent configuration file."
)
@click.option(
    "--mcp-config", default=None, help="Path to the mcp servers configuration file."
)
@click.option(
    "--console",
    is_flag=True,
    default=False,
    help="Use GUI interface instead of console",
)
def chat(provider, agent_config, mcp_config, console):
    """Start an interactive chat session with LLM"""
    try:
        # Only check environment variables if provider wasn't explicitly specified
        if provider is None:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider = "claude"
            elif os.getenv("GEMINI_API_KEY"):
                provider = "google"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("GROQ_API_KEY"):
                provider = "groq"
            elif os.getenv("DEEPINFRA_API_KEY"):
                provider = "deepinfra"
            else:
                raise ValueError(
                    "No LLM API key found. Please set either ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or DEEPINFRA_API_KEY"
                )
        services = setup_services(provider)

        if mcp_config:
            os.environ["MCP_CONFIG_PATH"] = mcp_config

        # Set up the agent system
        setup_agents(services, agent_config)

        # Create the message handler
        message_handler = MessageHandler(
            services["memory"], services["context_persistent"]
        )

        # Choose between GUI and console based on the --gui flag
        if not console:
            app = QApplication(sys.argv)
            chat_window = ChatWindow(message_handler)
            chat_window.show()
            sys.exit(app.exec())
        else:
            ui = ConsoleUI(message_handler)
            ui.start()
    except SystemExit:
        from swissknife.modules.mcpclient import MCPSessionManager

        MCPSessionManager.get_instance().cleanup()
    except Exception as e:
        print(traceback.format_exc())
        click.echo(f"❌ Error: {str(e)}", err=True)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--port", default=41241, help="Port to bind the server to")
@click.option("--base-url", default=None, help="Base URL for agent endpoints")
@click.option(
    "--provider",
    type=click.Choice(["claude", "groq", "openai", "google", "deepinfra"]),
    default=None,
    help="LLM provider to use (claude, groq, openai, google, or deepinfra)",
)
@click.option("--agent-config", default=None, help="Path to agent configuration file")
def a2a_server(host, port, base_url, provider, agent_config):
    """Start an A2A server exposing all SwissKnife agents"""
    try:
        # Set default base URL if not provided
        if not base_url:
            base_url = f"http://{host}:{port}"

        if provider is None:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider = "claude"
            elif os.getenv("GEMINI_API_KEY"):
                provider = "google"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("GROQ_API_KEY"):
                provider = "groq"
            elif os.getenv("DEEPINFRA_API_KEY"):
                provider = "deepinfra"
            else:
                raise ValueError(
                    "No LLM API key found. Please set either ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or DEEPINFRA_API_KEY"
                )
        services = setup_services(provider)

        # Set up agents from configuration
        setup_agents(services, agent_config)

        # Get agent manager
        agent_manager = AgentManager.get_instance()

        # Create and start server
        from swissknife.modules.a2a.server import A2AServer

        server = A2AServer(
            agent_manager=agent_manager, host=host, port=port, base_url=base_url
        )

        click.echo(f"Starting A2A server on {host}:{port}")
        click.echo(f"Available agents: {', '.join(agent_manager.agents.keys())}")
        server.start()
    except Exception as e:
        print(traceback.format_exc())
        click.echo(f"❌ Error: {str(e)}", err=True)


if __name__ == "__main__":
    cli()
