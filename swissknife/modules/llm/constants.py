from .types import Model

_ANTHROPIC_MODELS = [
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
]

_OPENAI_MODELS = [
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
        id="gpt-4o-mini",
        provider="openai",
        name="GPT-4o Mini",
        description="small, quick GPT model",
        capabilities=["tool_use", "vision"],
        input_token_price_1m=0.15,
        output_token_price_1m=0.6,
        default=True,
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
        id="o4-mini",
        provider="openai",
        name="GPT o4 mini",
        description="o4-mini is our latest small o-series model. It's optimized for fast, effective reasoning with exceptionally efficient performance in coding and visual tasks.",
        capabilities=["thinking", "tool_use", "vision"],
        input_token_price_1m=1.1,
        output_token_price_1m=4.4,
    ),
    Model(
        id="o3",
        provider="openai",
        name="GPT o3",
        description="a well-rounded and powerful model across domains. It sets a new standard for math, science, coding, and visual reasoning tasks. ",
        capabilities=["thinking", "tool_use", "vision"],
        input_token_price_1m=10.0,
        output_token_price_1m=40.0,
    ),
]

_GROQ_MODELS = [
    Model(
        id="compound-beta",
        provider="groq",
        name="Agentic Tooling model",
        description="Groq's first compound models with tooling",
        capabilities=["thinking"],
        input_token_price_1m=0.0,
        output_token_price_1m=0.0,
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
]

_GOOGLE_MODELS = [
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
        id="gemini-2.5-flash-preview-04-17",
        provider="google",
        name="Gemini 2.5 Flash Preview",
        description="Gemini 2.5 Flash is Google's first fully hybrid reasoning AI model, designed for high speed and cost-efficiency, allowing developers to toggle advanced reasoning on or off as needed.",
        capabilities=["tool_use", "vision", "thinking"],
        input_token_price_1m=0.15,
        output_token_price_1m=3.5,
    ),
    Model(
        id="gemini-2.5-pro-preview-03-25",
        provider="google",
        name="Gemini 2.5 Pro Thinking",
        description="Gemini 2.5 Pro with thinking",
        capabilities=["tool_use", "thinking", "vision"],
        input_token_price_1m=1.25,
        output_token_price_1m=2.5,
        default=True,
    ),
]

_DEEPINFRA_MODELS = [
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
        name="Llama 4 Maverick",
        description="The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding",
        capabilities=["text-generation", "tool_use"],
        input_token_price_1m=0.2,
        output_token_price_1m=0.6,
    ),
    Model(
        id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        provider="deepinfra",
        name="Llama 4 Scout",
        description="The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding",
        capabilities=["text-generation", "tool_use"],
        input_token_price_1m=0.08,
        output_token_price_1m=0.3,
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
        capabilities=["text-generation", "tool_use"],
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
AVAILABLE_MODELS = (
    _ANTHROPIC_MODELS
    + _OPENAI_MODELS
    + _GROQ_MODELS
    + _GOOGLE_MODELS
    + _DEEPINFRA_MODELS
)
