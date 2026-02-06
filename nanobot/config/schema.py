"""Configuration schema using Pydantic."""

from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class WhatsAppConfig(BaseModel):
    """WhatsApp channel configuration."""

    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"
    allow_from: list[str] = Field(default_factory=list)  # Allowed phone numbers

    @field_validator("allow_from", mode="before")
    @classmethod
    def coerce_allow_from(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(item) for item in value]
        return value


class TelegramConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""  # Bot token from @BotFather
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs or usernames

    @field_validator("allow_from", mode="before")
    @classmethod
    def coerce_allow_from(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(item) for item in value]
        return value


class ChannelsConfig(BaseModel):
    """Configuration for chat channels."""

    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class ModelSpecBase(BaseModel):
    """Model configuration without fallbacks."""

    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20


class ModelSpec(ModelSpecBase):
    """Model configuration with optional fallbacks."""

    fallbacks: list[ModelSpecBase] = Field(default_factory=list)

    @field_validator("fallbacks", mode="before")
    @classmethod
    def coerce_fallbacks(cls, value: object) -> object:
        if isinstance(value, list):
            updated: list[object] = []
            for item in value:
                if isinstance(item, str):
                    updated.append({"model": item})
                elif isinstance(item, dict):
                    # Convert camelCase to snake_case
                    converted = dict(item)
                    if "maxTokens" in converted:
                        converted["max_tokens"] = converted.pop("maxTokens")
                    if "maxToolIterations" in converted:
                        converted["max_tool_iterations"] = converted.pop("maxToolIterations")
                    updated.append(converted)
                else:
                    updated.append(item)
            return updated
        return value


class AgentDefaults(BaseModel):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: ModelSpec = Field(default_factory=ModelSpec)

    @field_validator("model", mode="before")
    @classmethod
    def coerce_model(cls, value: object) -> object:
        if isinstance(value, str):
            return {"model": value}
        return value


class AgentRole(BaseModel):
    """Role configuration for subagents."""

    model: ModelSpec | None = None
    tool_allow: list[str] = Field(default_factory=list)
    tool_deny: list[str] = Field(default_factory=list)
    max_tool_iterations: int | None = None

    @field_validator("model", mode="before")
    @classmethod
    def coerce_role_model(cls, value: object) -> object:
        if isinstance(value, str):
            return {"model": value}
        return value


class AgentsConfig(BaseModel):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    roles: dict[str, AgentRole] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""

    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    opencode: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)


class GatewayConfig(BaseModel):
    """Gateway/server configuration."""

    host: str = "127.0.0.1"
    port: int = 18790


class WebSearchConfig(BaseModel):
    """Web search tool configuration."""

    api_key: str = ""  # Brave Search API key
    max_results: int = 5


class WebToolsConfig(BaseModel):
    """Web tools configuration."""

    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ToolsConfig(BaseModel):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def get_api_key(self) -> str | None:
        """Get API key using model prefix or priority order."""
        provider = self._provider_for_model(self.agents.defaults.model)
        if provider is not None:
            return provider.api_key or None

        return (
            self.providers.openrouter.api_key
            or self.providers.anthropic.api_key
            or self.providers.openai.api_key
            or self.providers.opencode.api_key
            or self.providers.gemini.api_key
            or self.providers.zhipu.api_key
            or self.providers.groq.api_key
            or self.providers.vllm.api_key
            or None
        )

    def get_api_base(self) -> str | None:
        """Get API base URL for the selected provider."""
        provider = self._provider_for_model(self.agents.defaults.model)
        if provider is not None:
            if provider is self.providers.openrouter and provider.api_key:
                return provider.api_base or "https://openrouter.ai/api/v1"
            if provider is self.providers.opencode and provider.api_key:
                return provider.api_base or "https://opencode.ai/zen/v1"
            if provider is self.providers.zhipu and provider.api_key:
                return provider.api_base
            if provider is self.providers.vllm and provider.api_base:
                return provider.api_base
            if provider.api_base:
                return provider.api_base
            return None

        if self.providers.openrouter.api_key:
            return self.providers.openrouter.api_base or "https://openrouter.ai/api/v1"
        if self.providers.opencode.api_key:
            return self.providers.opencode.api_base or "https://opencode.ai/zen/v1"
        if self.providers.zhipu.api_key:
            return self.providers.zhipu.api_base
        if self.providers.vllm.api_base:
            return self.providers.vllm.api_base
        return None

    def _provider_for_model(self, model: str | ModelSpec | None) -> ProviderConfig | None:
        model_id = model.model if isinstance(model, ModelSpec) else model
        if not model_id or "/" not in model_id:
            return None
        provider_id = model_id.split("/", 1)[0].lower()
        return {
            "openrouter": self.providers.openrouter,
            "anthropic": self.providers.anthropic,
            "openai": self.providers.openai,
            "opencode": self.providers.opencode,
            "gemini": self.providers.gemini,
            "zhipu": self.providers.zhipu,
            "zai": self.providers.zhipu,
            "groq": self.providers.groq,
            "vllm": self.providers.vllm,
        }.get(provider_id)

    class Config:
        env_prefix = "NANOBOT_"
        env_nested_delimiter = "__"
