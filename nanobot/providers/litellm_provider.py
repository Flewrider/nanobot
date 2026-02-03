"""LiteLLM provider implementation for multi-provider support."""

import os
from typing import Any

import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.model_registry import get_model_registry


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.

    Supports OpenRouter, Anthropic, OpenAI, Gemini, and many other providers through
    a unified interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        provider_keys: dict[str, str] | None = None,
        provider_bases: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.provider_keys = provider_keys or {}
        self.provider_bases = provider_bases or {}

        # Detect OpenRouter by api_key prefix or explicit api_base
        self.is_openrouter = bool(
            (api_key and api_key.startswith("sk-or-"))
            or (api_base and "openrouter" in api_base)
            or self.provider_bases.get("openrouter")
            or self.provider_keys.get("openrouter")
        )

        # Detect OpenCode Zen by explicit api_base or model prefix
        self.is_opencode = bool(api_base) and "opencode.ai/zen" in api_base
        if default_model.startswith("opencode/"):
            self.is_opencode = True
        if self.is_opencode and not api_base:
            api_base = "https://opencode.ai/zen/v1"
            self.api_base = api_base

        # Track if using custom endpoint (vLLM, etc.)
        self.is_vllm = bool(api_base) and not self.is_openrouter and not self.is_opencode
        if self.provider_bases.get("vllm"):
            self.is_vllm = True

        # Configure LiteLLM based on provider
        if self.provider_keys:
            self._apply_provider_env(self.provider_keys)
        elif api_key:
            if self.is_openrouter:
                # OpenRouter mode - set key
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif self.is_vllm:
                # vLLM/custom endpoint - uses OpenAI-compatible API
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in default_model:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif "openai" in default_model or "gpt" in default_model:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif "gemini" in default_model.lower():
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            elif "zhipu" in default_model or "glm" in default_model or "zai" in default_model:
                os.environ.setdefault("ZHIPUAI_API_KEY", api_key)
            elif "groq" in default_model:
                os.environ.setdefault("GROQ_API_KEY", api_key)

        if api_base:
            litellm.api_base = api_base

        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = model or self.default_model

        self._apply_model_env(model)

        # For OpenRouter, prefix model name if not already prefixed
        if self.is_openrouter and not model.startswith("openrouter/"):
            model = f"openrouter/{model}"
            registry = get_model_registry()
            await registry.ensure_provider_cache("openrouter")

        # For OpenCode Zen, use OpenAI-compatible provider with Zen base URL
        if model.startswith("opencode/"):
            model_id = model.split("/", 1)[1]
            registry = get_model_registry()
            opencode_provider = await registry.get_opencode_provider(model_id)
            opencode_key = self.provider_keys.get("opencode") or self.api_key
            if opencode_provider == "anthropic":
                model = f"anthropic/{model_id}"
                if opencode_key:
                    os.environ["ANTHROPIC_API_KEY"] = opencode_key
                self.api_base = self.provider_bases.get("opencode") or "https://opencode.ai/zen"
            else:
                if opencode_provider not in {"openai", "anthropic"}:
                    print(
                        f"Warning: Opencode model {model_id} uses {opencode_provider}; "
                        "defaulting to OpenAI-compatible routing."
                    )
                model = f"openai/{model_id}"
                if opencode_key:
                    os.environ["OPENAI_API_KEY"] = opencode_key
                if not self.api_base:
                    self.api_base = (
                        self.provider_bases.get("opencode") or "https://opencode.ai/zen/v1"
                    )

        # For Zhipu/Z.ai, ensure prefix is present
        # Handle cases like "glm-4.7-flash" -> "zhipu/glm-4.7-flash"
        if ("glm" in model.lower() or "zhipu" in model.lower()) and not (
            model.startswith("zhipu/")
            or model.startswith("zai/")
            or model.startswith("openrouter/")
        ):
            model = f"zhipu/{model}"

        # For vLLM, use hosted_vllm/ prefix per LiteLLM docs
        # Convert openai/ prefix to hosted_vllm/ if user specified it
        if self.is_vllm:
            model = f"hosted_vllm/{model}"

        # For Gemini, ensure gemini/ prefix if not already present
        if "gemini" in model.lower() and not model.startswith("gemini/"):
            model = f"gemini/{model}"

        if model.startswith("openai/"):
            registry = get_model_registry()
            await registry.ensure_provider_cache("openai", api_key=self.api_key)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Pass api_base directly for custom endpoints (vLLM, etc.)
        api_base = self.api_base
        if model.startswith("openrouter/"):
            api_base = self.provider_bases.get("openrouter") or api_base
        elif self.is_vllm:
            api_base = self.provider_bases.get("vllm") or api_base

        if api_base and (not model.startswith("anthropic/") or self.is_opencode):
            kwargs["api_base"] = api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    def _apply_provider_env(self, provider_keys: dict[str, str]) -> None:
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "zhipu": "ZHIPUAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "vllm": "OPENAI_API_KEY",
        }
        for provider, key in provider_keys.items():
            env_var = env_map.get(provider)
            if env_var and key:
                os.environ.setdefault(env_var, key)

    def _apply_model_env(self, model: str) -> None:
        provider = model.split("/", 1)[0].lower() if "/" in model else ""
        key_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "opencode": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "zhipu": "ZHIPUAI_API_KEY",
            "zai": "ZHIPUAI_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        key = self.provider_keys.get(provider)
        env_var = key_map.get(provider)
        if env_var and key:
            os.environ[env_var] = key

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    import json

                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
