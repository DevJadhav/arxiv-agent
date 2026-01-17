"""Unified LLM service supporting multiple providers."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
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


class TokenTracker:
    """Tracks token usage validation."""
    
    def __init__(self, path: Path):
        self.path = path
        self._load()
        
    def _load(self):
        if self.path.exists():
            try:
                import json
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"daily": {}, "total": {}}
        else:
            self.data = {"daily": {}, "total": {}}
            
    def _save(self):
        try:
            import json
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save token usage: {e}")
            
    def track(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        import datetime
        today = datetime.date.today().isoformat()
        
        # Init keys
        if today not in self.data["daily"]:
            self.data["daily"][today] = 0
        
        total_tokens = input_tokens + output_tokens
        self.data["daily"][today] += total_tokens
        
        # Track per-model totals
        key = f"{provider}/{model}"
        if key not in self.data["total"]:
            self.data["total"][key] = 0
        self.data["total"][key] += total_tokens
        
        self._save()
        
    def check_daily_limit(self, limit: int) -> bool:
        import datetime
        today = datetime.date.today().isoformat()
        usage = self.data["daily"].get(today, 0)
        return usage >= limit


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
        self._ollama_client = None
        
        self.max_tokens = self.settings.llm.max_tokens
        self.temperature = self.settings.llm.temperature
        
        # Token Tracking
        self.tracker = TokenTracker(self.settings.data_dir / "usage.json")
        
        # Conversation history
        self.history: list[dict] = []
        
    def check_context_window(self, prompt: str, system_prompt: str | None = None) -> None:
        """Check if request fits within context window."""
        # Rough estimation (4 chars per token) to avoid heavy deps
        estimated_tokens = len(prompt) / 4
        if system_prompt:
            estimated_tokens += len(system_prompt) / 4
            
        # Add history estimation
        for msg in self.history:
            estimated_tokens += len(msg["content"]) / 4
            
        limit = 120_000 # Default safe limit for most modern models
        if "128k" in self.model or "opus" in self.model or "gpt-4" in self.model:
             limit = 120_000
        elif "flash" in self.model or "sonnet" in self.model:
             limit = 180_000 
             
        if estimated_tokens > limit:
             logger.warning(
                 f"Request size ({int(estimated_tokens)} tokens) is approaching model limit (~{limit}). "
                 "Truncation or errors may occur."
             )
    
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

    def _get_ollama(self):
        """Get or create Ollama client."""
        if self._ollama_client is None:
            try:
                import ollama
                self._ollama_client = ollama.Client(host=self.settings.llm.ollama_base_url)
            except ImportError as e:
                raise ImportError(
                    "Ollama package not installed. Install with: pip install ollama "
                    "or install arxiv-agent with local-llm extras: pip install arxiv-agent[local-llm]"
                ) from e
        return self._ollama_client
    
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
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt, use_history, temperature, max_tokens)
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
        
        # Guardrails
        self.check_context_window(prompt, system_prompt)
        
        # Check limits
        if self.settings.guardrails.warn_on_costly_request:
            if self.tracker.check_daily_limit(self.settings.guardrails.max_daily_tokens):
                logger.warning(
                    f"Daily token limit ({self.settings.guardrails.max_daily_tokens}) exceeded! "
                    "Proceeding, but be aware of costs."
                )

        response = client.messages.create(**kwargs)
        
        content = response.content[0].text
        
        # Update history
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
            
        # Track usage
        self.tracker.track(
            "anthropic", 
            self.model, 
            response.usage.input_tokens, 
            response.usage.output_tokens
        )
        
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
        
        # Guardrails
        self.check_context_window(prompt, system_prompt)

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
            
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Track usage
        self.tracker.track("openai", self.model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using Ollama."""
        client = self._get_ollama()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})

        # Guardrails
        self.check_context_window(prompt, system_prompt)

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = self.temperature
        if max_tokens:
            options["num_predict"] = max_tokens
        else:
            options["num_predict"] = self.max_tokens

        response = client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )

        content = response["message"]["content"]

        # Update history
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})

        # Track usage (Ollama provides token counts)
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        self.tracker.track("ollama", self.model, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        elif self.provider == "ollama":
            return await self._agenerate_ollama(prompt, system_prompt, use_history, temperature, max_tokens)
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
            logger.warning(f"Failed to parse JSON response: {e}. Content: {content}")
            
            if self.settings.guardrails.enable_json_repair:
                logger.info("Attempting to repair JSON with LLM...")
                repair_prompt = (
                    f"The following JSON is invalid:\n{content}\n\n"
                    f"Error: {str(e)}\n\n"
                    "Please fix the JSON and return ONLY the valid JSON object."
                )
                try:
                    repair_response = await self.agenerate(
                        prompt=repair_prompt,
                        system_prompt="You are a JSON repair tool. Return only valid JSON.",
                        temperature=0.0,
                    )
                    # Clean repaired content
                    repaired_content = repair_response.content.strip()
                    if repaired_content.startswith("```json"):
                        repaired_content = repaired_content[7:]
                    elif repaired_content.startswith("```"):
                        repaired_content = repaired_content[3:]
                    if repaired_content.endswith("```"):
                        repaired_content = repaired_content[:-3]
                        
                    return json.loads(repaired_content.strip())
                except Exception as repair_e:
                    logger.error(f"JSON repair failed: {repair_e}")
                    raise ValueError(f"LLM failed to generate valid JSON after repair: {str(e)}")
            
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

        # Guardrails
        self.check_context_window(prompt, system_prompt)
        
        response = await client.messages.create(**kwargs)
        
        content = response.content[0].text
        
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
            
        # Track usage
        self.tracker.track(
            "anthropic", 
            self.model, 
            response.usage.input_tokens, 
            response.usage.output_tokens
        )
        
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
        
        # Guardrails
        self.check_context_window(prompt, system_prompt)

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
            
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
            
        # Track usage
        self.tracker.track("openai", self.model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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

    async def _agenerate_ollama(
        self,
        prompt: str,
        system_prompt: str | None,
        use_history: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate using Ollama async."""
        try:
            import ollama
            client = ollama.AsyncClient(host=self.settings.llm.ollama_base_url)
        except ImportError as e:
            raise ImportError(
                "Ollama package not installed. Install with: pip install ollama "
                "or install arxiv-agent with local-llm extras: pip install arxiv-agent[local-llm]"
            ) from e

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})

        # Guardrails
        self.check_context_window(prompt, system_prompt)

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = self.temperature
        if max_tokens:
            options["num_predict"] = max_tokens
        else:
            options["num_predict"] = self.max_tokens

        response = await client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )

        content = response["message"]["content"]

        # Update history
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})

        # Track usage (Ollama provides token counts)
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        self.tracker.track("ollama", self.model, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
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
        elif self.provider == "ollama":
            async for chunk in self._stream_ollama(prompt, system_prompt):
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

    async def _stream_ollama(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama."""
        try:
            import ollama
            client = ollama.AsyncClient(host=self.settings.llm.ollama_base_url)
        except ImportError as e:
            raise ImportError(
                "Ollama package not installed. Install with: pip install ollama "
                "or install arxiv-agent with local-llm extras: pip install arxiv-agent[local-llm]"
            ) from e

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        async for chunk in await client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
        ):
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
    
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
        elif self.provider == "ollama":
            return self._list_ollama_models()
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

    def _list_ollama_models(self) -> list[str]:
        """List Ollama models available on the local server."""
        try:
            client = self._get_ollama()
            response = client.list()
            # Handle both dict response and ListResponse object from ollama library
            if hasattr(response, 'models'):
                # ListResponse object with Model objects
                return sorted([m.model for m in response.models], reverse=True)
            else:
                # Dict response (older API)
                return sorted([m["name"] for m in response.get("models", [])], reverse=True)
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            # Fallback to default models if server is not running or list fails
            return [
                self.settings.llm.ollama_model_advanced,
                self.settings.llm.ollama_model,
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
