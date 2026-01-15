"""Unified LLM service supporting multiple providers."""

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Any

from loguru import logger

from arxiv_agent.config.settings import get_settings, LLMProviderType
from arxiv_agent.config.keys import get_key_storage


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    
    content: str
    model: str
    provider: LLMProviderType
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None


class LLMService:
    """Unified LLM service supporting Anthropic, OpenAI, and Gemini.
    
    Provides a consistent interface across all providers with support for:
    - Synchronous and asynchronous generation
    - Streaming responses
    - Conversation history
    - Per-agent model configuration
    """
    
    def __init__(
        self,
        provider: LLMProviderType | None = None,
        model: str | None = None,
        agent: str | None = None,
    ):
        """Initialize LLM service.
        
        Args:
            provider: Override default provider
            model: Override default model
            agent: Agent name for per-agent config (analyzer, chat, code, digest)
        """
        self.settings = get_settings()
        self.key_storage = get_key_storage()
        
        # Determine provider and model
        if agent:
            self.provider, self.model = self.settings.llm.get_agent_config(agent)
        else:
            self.provider = provider or self.settings.llm.default_provider
            self.model = model or self.settings.llm.get_provider_model(self.provider)
        
        # Override if explicitly specified
        if provider:
            self.provider = provider
        if model:
            self.model = model
        
        # Initialize clients lazily
        self._anthropic_client = None
        self._openai_client = None
        self._gemini_client = None
        
        self.max_tokens = self.settings.llm.max_tokens
        self.temperature = self.settings.llm.temperature
        
        # Conversation history
        self.history: list[dict] = []
    
    def _get_api_key(self, provider: LLMProviderType) -> str:
        """Get API key for provider."""
        key = self.key_storage.get_key(provider)
        if not key:
            raise ValueError(
                f"No API key configured for {provider}. "
                f"Use 'arxiv-agent config provider setup {provider}' to configure."
            )
        return key
    
    def _get_anthropic(self):
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self._get_api_key("anthropic")
            )
        return self._anthropic_client
    
    def _get_openai(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=self._get_api_key("openai")
            )
        return self._openai_client
    
    def _get_gemini(self):
        """Get or create Gemini client."""
        if self._gemini_client is None:
            import google.generativeai as genai
            genai.configure(api_key=self._get_api_key("gemini"))
            self._gemini_client = genai
        return self._gemini_client
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_history: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response synchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        
        Returns:
            LLMResponse with generated content
        """
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, use_history, temperature, max_tokens)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, use_history, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, system_prompt, use_history, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using Anthropic Claude."""
        client = self._get_anthropic()
        
        messages = []
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }
        # Anthropic doesn't support temperature in all models/clients easily or check if supported
        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.temperature
            
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = client.messages.create(**kwargs)
        
        content = response.content[0].text
        
        # Update history
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using OpenAI."""
        client = self._get_openai()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            messages=messages,
        )
        
        content = response.choices[0].message.content
        
        # Update history
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
        )
    
    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using Google Gemini."""
        genai = self._get_gemini()
        
        # Build prompt with system instruction
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Include history
        if use_history and self.history:
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in self.history
            ])
            full_prompt = f"{history_text}\n\nUSER: {full_prompt}"
        
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            ),
        )
        
        content = response.text
        
        # Update history
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            finish_reason="stop",
        )
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_history: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        
        Returns:
            LLMResponse with generated content
        """
        if self.provider == "anthropic":
            return await self._agenerate_anthropic(prompt, system_prompt, use_history, temperature, max_tokens)
        elif self.provider == "openai":
            return await self._agenerate_openai(prompt, system_prompt, use_history, temperature, max_tokens)
        elif self.provider == "gemini":
            return await self._agenerate_gemini(prompt, system_prompt, use_history, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Generate structured JSON output.
        
        Args:
            prompt: User prompt
            output_schema: JSON Schema dict describing output format
            system_prompt: Optional system prompt
            temperature: Lower temperature for structured output
        
        Returns:
            Parsed JSON dictionary
        """
        import json
        
        schema_str = json.dumps(output_schema, indent=2)
        
        full_system_prompt = (
            f"{system_prompt or 'You are a helpful AI assistant.'}\n\n"
            f"You MUST return a valid JSON object adhering to the following schema:\n"
            f"```json\n"
            f"{schema_str}\n"
            f"```\n"
            f"Do not include any text before or after the JSON. "
            f"Ensure the JSON is valid and can be parsed by Python's json.loads()."
        )
        
        response = await self.agenerate(
            prompt=prompt,
            system_prompt=full_system_prompt,
            temperature=temperature,
            use_history=False,
        )
        
        content = response.content.strip()
        
        # Clean up code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nContent: {content}")
            raise ValueError(f"LLM failed to generate valid JSON: {str(e)}")

    async def _agenerate_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using Anthropic Claude async."""
        import anthropic
        
        client = anthropic.AsyncAnthropic(
            api_key=self._get_api_key("anthropic")
        )
        
        messages = []
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.temperature
        
        response = await client.messages.create(**kwargs)
        
        content = response.content[0].text
        
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )
    
    async def _agenerate_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using OpenAI async."""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self._get_api_key("openai")
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            messages=messages,
        )
        
        content = response.choices[0].message.content
        
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
        )
    
    async def _agenerate_gemini(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using Gemini async."""
        # Gemini doesn't have official async, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_gemini(prompt, system_prompt, use_history, temperature, max_tokens)
        )
    
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Yields:
            Response chunks as they arrive
        """
        if self.provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt, system_prompt):
                yield chunk
        elif self.provider == "openai":
            async for chunk in self._stream_openai(prompt, system_prompt):
                yield chunk
        elif self.provider == "gemini":
            async for chunk in self._stream_gemini(prompt, system_prompt):
                yield chunk
    
    async def _stream_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> AsyncGenerator[str, None]:
        """Stream from Anthropic."""
        import anthropic
        
        client = anthropic.AsyncAnthropic(
            api_key=self._get_api_key("anthropic")
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def _stream_openai(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> AsyncGenerator[str, None]:
        """Stream from OpenAI."""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self._get_api_key("openai")
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = await client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _stream_gemini(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> AsyncGenerator[str, None]:
        """Stream from Gemini."""
        genai = self._get_gemini()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
    
    def set_history(self, history: list[dict]) -> None:
        """Set conversation history.
        
        Args:
            history: List of messages with 'role' and 'content' keys
        """
        self.history = history.copy()


    def list_models(self) -> list[str]:
        """List available models for the current provider."""
        if self.provider == "anthropic":
            return self._list_anthropic_models()
        elif self.provider == "openai":
            return self._list_openai_models()
        elif self.provider == "gemini":
            return self._list_gemini_models()
        else:
            return []

    def _list_anthropic_models(self) -> list[str]:
        """List Anthropic models."""
        # Anthropic now supports listing models
        try:
            client = self._get_anthropic()
            models = client.models.list()
            return sorted([m.id for m in models.data], reverse=True)
        except Exception as e:
            logger.warning(f"Failed to list Anthropic models: {e}")
            # Fallback to known models if API fails or key is invalid/missing
            return [
                self.settings.llm.anthropic_model_advanced,
                self.settings.llm.anthropic_model,
                self.settings.llm.anthropic_model_fast,
            ]

    def _list_openai_models(self) -> list[str]:
        """List OpenAI models."""
        try:
            client = self._get_openai()
            # Set a timeout for the list operation to prevent hangs
            models = client.models.list(timeout=10.0)
            # Filter for chat models (gpt-*)
            return sorted(
                [m.id for m in models.data if m.id.startswith("gpt-") and "instruct" not in m.id and "realtime" not in m.id],
                reverse=True
            )
        except Exception as e:
            logger.warning(f"Failed to list OpenAI models: {e}")
            return [
                self.settings.llm.openai_model_advanced,
                self.settings.llm.openai_model,
            ]

    def _list_gemini_models(self) -> list[str]:
        """List Gemini models."""
        try:
            genai = self._get_gemini()
            models = genai.list_models()
            return sorted(
                [m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods],
                reverse=True
            )
        except Exception as e:
            logger.warning(f"Failed to list Gemini models: {e}")
            return [
                self.settings.llm.gemini_model_advanced,
                self.settings.llm.gemini_model,
            ]


def get_llm_service(
    provider: LLMProviderType | None = None,
    model: str | None = None,
    agent: str | None = None,
) -> LLMService:
    """Create an LLM service instance.
    
    Args:
        provider: Override default provider
        model: Override default model
        agent: Agent name for per-agent config
    
    Returns:
        Configured LLMService
    """
    return LLMService(provider=provider, model=model, agent=agent)
