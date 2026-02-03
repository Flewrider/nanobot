"""Model registry with cached provider metadata."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

OPENCODE_MODEL_URL = "https://opencode.ai/zen/v1/models"
OPENCODE_MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"


class ModelRegistry:
    def __init__(self, cache_dir: Path, ttl_seconds: int) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_opencode_provider(self, model_id: str) -> str:
        data = await self._get_provider_data("opencode")
        provider = data.get(model_id)
        if provider:
            return provider
        return "openai"

    async def ensure_provider_cache(self, provider: str, api_key: str | None = None) -> None:
        await self._get_provider_data(provider, api_key=api_key)

    async def _get_provider_data(self, provider: str, api_key: str | None = None) -> dict[str, str]:
        cached = self._cache.get(provider) or self._load_cache(provider)
        if cached and self._is_cache_valid(cached):
            return cached.get("models", {})

        lock = self._locks.setdefault(provider, asyncio.Lock())
        async with lock:
            cached = self._cache.get(provider) or self._load_cache(provider)
            if cached and self._is_cache_valid(cached):
                return cached.get("models", {})

            data = await self._fetch_provider_data(provider, api_key=api_key)
            if data:
                self._cache[provider] = data
                self._save_cache(provider, data)
                return data.get("models", {})

            if cached:
                return cached.get("models", {})
        return {}

    async def _fetch_provider_data(
        self, provider: str, api_key: str | None = None
    ) -> dict[str, Any] | None:
        if provider == "opencode":
            return await self._fetch_opencode_models()
        if provider == "openrouter":
            return await self._fetch_openrouter_models()
        if provider == "openai":
            return await self._fetch_openai_models(api_key=api_key)
        return None

    async def _fetch_opencode_models(self) -> dict[str, Any] | None:
        ttl = self.ttl_seconds
        timeout = httpx.Timeout(5.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(OPENCODE_MODELS_DEV_URL)
                response.raise_for_status()
                payload = response.json()
                provider_payload = payload.get("opencode") or {}
                models_payload = provider_payload.get("models") or {}
                models = self._parse_opencode_models(models_payload)
                if models:
                    return {"fetched_at": int(time.time()), "ttl": ttl, "models": models}
            except Exception:
                pass

            try:
                response = await client.get(OPENCODE_MODEL_URL)
                response.raise_for_status()
                payload = response.json()
                models = {
                    item.get("id"): "openai"
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                }
                if models:
                    return {"fetched_at": int(time.time()), "ttl": ttl, "models": models}
            except Exception:
                return None

        return None

    async def _fetch_openrouter_models(self) -> dict[str, Any] | None:
        ttl = self.ttl_seconds
        timeout = httpx.Timeout(5.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(OPENROUTER_MODELS_URL)
                response.raise_for_status()
                payload = response.json()
                models = {
                    item.get("id"): "openrouter"
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                }
                if models:
                    return {"fetched_at": int(time.time()), "ttl": ttl, "models": models}
            except Exception:
                return None
        return None

    async def _fetch_openai_models(self, api_key: str | None) -> dict[str, Any] | None:
        if not api_key:
            return None
        ttl = self.ttl_seconds
        timeout = httpx.Timeout(5.0, connect=5.0)
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(OPENAI_MODELS_URL, headers=headers)
                response.raise_for_status()
                payload = response.json()
                models = {
                    item.get("id"): "openai"
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                }
                if models:
                    return {"fetched_at": int(time.time()), "ttl": ttl, "models": models}
            except Exception:
                return None
        return None

    def _parse_opencode_models(self, models_payload: dict[str, Any]) -> dict[str, str]:
        models: dict[str, str] = {}
        for model_id, meta in models_payload.items():
            if not isinstance(meta, dict):
                continue
            provider_meta = meta.get("provider") or {}
            npm = provider_meta.get("npm", "")
            provider = "openai"
            npm_lower = str(npm).lower()
            if "anthropic" in npm_lower:
                provider = "anthropic"
            elif "google" in npm_lower or "gemini" in npm_lower:
                provider = "gemini"
            elif "openai" in npm_lower or "openai-compatible" in npm_lower:
                provider = "openai"
            models[model_id] = provider
        return models

    def _cache_path(self, provider: str) -> Path:
        return self.cache_dir / f"{provider}.json"

    def _load_cache(self, provider: str) -> dict[str, Any] | None:
        path = self._cache_path(provider)
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, dict):
                self._cache[provider] = data
                return data
        except Exception:
            return None
        return None

    def _save_cache(self, provider: str, data: dict[str, Any]) -> None:
        path = self._cache_path(provider)
        try:
            path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        except Exception:
            return

    def _is_cache_valid(self, data: dict[str, Any]) -> bool:
        fetched_at = data.get("fetched_at")
        if not isinstance(fetched_at, int):
            return False
        ttl = data.get("ttl")
        if not isinstance(ttl, int):
            ttl = self.ttl_seconds
        return (time.time() - fetched_at) < ttl


_REGISTRY: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY:
        return _REGISTRY

    ttl_hours = int(os.getenv("NANOBOT_MODEL_CACHE_TTL_HOURS", "24"))
    ttl_seconds = max(ttl_hours, 1) * 60 * 60
    cache_dir = Path.home() / ".nanobot" / "cache" / "models"
    _REGISTRY = ModelRegistry(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
    return _REGISTRY
