from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Model:
    """Model metadata class."""

    id: str
    provider: str
    name: str
    description: str
    capabilities: List[str]
    default: bool = False
    input_token_price_1m: float = 0.0
    output_token_price_1m: float = 0.0


class ModelRegistry:
    """Registry for available LLM models."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ModelRegistry."""
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def __init__(self):
        """Initialize the model registry with default models."""
        if ModelRegistry._instance is not None:
            raise RuntimeError(
                "ModelRegistry is a singleton. Use get_instance() instead."
            )

        self.models: Dict[str, Model] = {}
        self.current_model: Optional[Model] = None
        self._initialize_default_models()

    @classmethod
    def get_model_capabilities(cls, mode_id):
        registry = ModelRegistry.get_instance()
        model = registry.get_model(mode_id)
        if not model:
            return []
        return model.capabilities

    def _initialize_default_models(self):
        """Initialize the registry with default models."""
        default_models = [
            Model(
                id="claude-3-7-sonnet-latest",
                provider="claude",
                name="Claude 3.7 Sonnet",
                description="Anthropic's most powerful model with advanced reasoning",
                capabilities=["thinking", "tool_use", "vision"],
                default=True,
                input_token_price_1m=3.0,
                output_token_price_1m=15.0,
            ),
            Model(
                id="claude-3-5-sonnet-latest",
                provider="claude",
                name="Claude 3.5 Sonnet",
                description="Anthropic's Claude 3.5 Sonnet model - balanced performance and capabilities",
                capabilities=["tool_use", "vision"],
                input_token_price_1m=3.0,
                output_token_price_1m=15.0,
            ),
            Model(
                id="claude-3-5-haiku-latest",
                provider="claude",
                name="Claude 3.5 Haiku",
                description="Anthropic's fastest model",
                capabilities=["tool_use", "vision"],
                input_token_price_1m=0.8,
                output_token_price_1m=4.0,
            ),
            Model(
                id="gpt-4o",
                provider="openai",
                name="GPT-4o",
                description="Fast, intelligent, flexible GPT model",
                capabilities=["tool_use", "vision"],
                input_token_price_1m=2.5,
                output_token_price_1m=10.0,
            ),
            Model(
                id="gpt-4.1",
                provider="openai",
                name="GPT-4.1",
                description="Flagship model for complex tasks. It is well suited for problem solving across domains",
                capabilities=["tool_use", "vision"],
                input_token_price_1m=2,
                output_token_price_1m=8,
                default=True,
            ),
            Model(
                id="o3-mini",
                provider="openai",
                name="GPT o3 mini",
                description="Fast, flexible, intelligent reasoning model",
                capabilities=["thinking"],
                input_token_price_1m=1.1,
                output_token_price_1m=4.4,
            ),
            Model(
                id="deepseek-r1-distill-llama-70b",
                provider="groq",
                name="DeepSeek R1 Distill",
                description="DeepSeek's powerful model optimized for Groq",
                capabilities=["thinking", "tool_use"],
                input_token_price_1m=0.75,
                output_token_price_1m=0.99,
            ),
            Model(
                id="deepseek-r1-distill-qwen-32b",
                provider="groq",
                name="DeepSeek R1 Distill Qwen 32b",
                description="DeepSeek's powerful model optimized for Groq",
                capabilities=["thinking", "tool_use"],
                input_token_price_1m=0.69,
                output_token_price_1m=0.69,
            ),
            Model(
                id="llama-3.3-70b-versatile",
                provider="groq",
                name="Llama 3.3 70B",
                description="Meta's Llama 3 70B model optimized for Groq",
                capabilities=["tool_use"],
                input_token_price_1m=0.59,
                output_token_price_1m=0.79,
            ),
            Model(
                id="meta-llama/llama-4-scout-17b-16e-instruct",
                provider="groq",
                name="Llama 4 Scout",
                description="The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding.",
                capabilities=["tool_use", "vision"],
                input_token_price_1m=0.11,
                output_token_price_1m=0.34,
            ),
            Model(
                id="meta-llama/llama-4-maverick-17b-128e-instruct",
                provider="groq",
                name="Llama 4 Maverick",
                description="The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding.",
                capabilities=["tool_use"],
                input_token_price_1m=0.5,
                output_token_price_1m=0.77,
            ),
            Model(
                id="qwen-qwq-32b",
                provider="groq",
                name="QwQ 32B",
                description="SLM from Alibaba",
                capabilities=["thinking", "tool_use"],
                default=False,
                input_token_price_1m=0.29,
                output_token_price_1m=0.39,
            ),
            Model(
                id="qwen-2.5-32b",
                provider="groq",
                name="Qwen 32B Instruct",
                description="the latest series of Qwen large language models",
                capabilities=["thinking", "tool_use"],
                default=True,
                input_token_price_1m=0.79,
                output_token_price_1m=0.79,
            ),
            Model(
                id="gemini-2.0-flash",
                provider="google",
                name="Gemini 2.0 Flash",
                description="Gemini 2.0 Flash is a powerful language model from Google, designed for both text and visual inputs.",
                capabilities=["tool_use", "vision"],
                input_token_price_1m=0.1,
                output_token_price_1m=0.4,
            ),
            Model(
                id="gemini-2.5-pro-preview-03-25",
                provider="google",
                name="Gemini 2.5 Pro Thinking",
                description="Gemini 2.5 Pro with thinking",
                capabilities=["tool_use", "thinking"],
                input_token_price_1m=1.25,
                output_token_price_1m=2.5,
                default=True,
            ),
            Model(
                id="meta-llama/Llama-3.3-70B-Instruct",
                provider="deepinfra",
                name="Llama 3.3 70B Instruct",
                description="Llama 3.3-70B is a multilingual LLM trained on a massive dataset of 15 trillion tokens, fine-tuned for instruction-following and conversational dialogue",
                capabilities=["tool_use", "text-generation"],
                input_token_price_1m=0.23,
                output_token_price_1m=0.40,
            ),
            Model(
                id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                provider="deepinfra",
                name="Llama 4 17B - 128E Instruct",
                description="The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding",
                capabilities=["text-generation"],
                input_token_price_1m=0.2,
                output_token_price_1m=0.6,
            ),
            Model(
                id="google/gemma-3-27b-it",
                provider="deepinfra",
                name="Gemma 3 27B",
                description="Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models",
                capabilities=["text-generation"],
                input_token_price_1m=0.1,
                output_token_price_1m=0.2,
            ),
            Model(
                id="deepseek-ai/DeepSeek-V3-0324",
                provider="deepinfra",
                name="Deepseek v3 0324",
                description="DeepSeek-V3-0324, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token, an improved iteration over DeepSeek-V3",
                capabilities=["text-generation"],
                input_token_price_1m=0.4,
                output_token_price_1m=0.89,
            ),
            Model(
                id="microsoft/Phi-4-multimodal-instruct",
                provider="deepinfra",
                name="Phi 4",
                description="Phi-4-multimodal-instruct is a lightweight open multimodal foundation model that leverages the language, vision, and speech research and datasets used for Phi-3.5 and 4.0 models.",
                capabilities=["text-generation", "vision"],
                input_token_price_1m=0.05,
                output_token_price_1m=0.1,
            ),
            Model(
                id="Qwen/Qwen2.5-72B-Instruct",
                provider="deepinfra",
                name="Qwen 2.5 72B Instruct",
                description="Qwen2.5 is a model pretrained on a large-scale dataset of up to 18 trillion tokens, offering significant improvements in knowledge, coding, mathematics, and instruction following compared to its predecessor Qwen2",
                capabilities=["text-generation", "vision", "tool_use"],
                input_token_price_1m=0.05,
                output_token_price_1m=0.1,
            ),
        ]

        for model in default_models:
            self.register_model(model)

        # Set the default model
        for model in default_models:
            if model.default:
                self.current_model = model
                break

    def register_model(self, model: Model):
        """
        Register a model in the registry.

        Args:
            model: The model to register
        """
        self.models[model.id] = model

    def get_model(self, model_id: str) -> Optional[Model]:
        """
        Get a model by ID.

        Args:
            model_id: The model ID

        Returns:
            The model if found, None otherwise
        """
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: str) -> List[Model]:
        """
        Get all models for a specific provider.

        Args:
            provider: The provider name

        Returns:
            List of models for the provider
        """
        return [model for model in self.models.values() if model.provider == provider]

    def set_current_model(self, model_id: str) -> bool:
        """
        Set the current model by ID.

        Args:
            model_id: The model ID

        Returns:
            True if successful, False otherwise
        """
        model = self.get_model(model_id)
        if model:
            self.current_model = model
            return True
        return False

    def get_current_model(self) -> Optional[Model]:
        """
        Get the current model.

        Returns:
            The current model if set, None otherwise
        """
        return self.current_model
