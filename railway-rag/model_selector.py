"""
Model Selector for intelligent LLM fallback with rotation and multi-key support.
"""
import time
import logging
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from groq import RateLimitError, APIStatusError, APITimeoutError
from google.api_core.exceptions import GoogleAPIError
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

logger = logging.getLogger(__name__)

class LLMExhaustedError(Exception):
    """Raised when all available LLM providers are exhausted"""
    pass

class ModelSelector:
    """
    Intelligent model selector that rotates through available models and API keys
    based on quotas, rate limits, and errors.

    Rotation: primary provider switches every 2 queries (Gemini <-> Groq).
    Key rotation: within a provider, keys are tried sequentially on quota errors.
    """

    def __init__(
        self,
        gemini_models: List[str],
        groq_models: List[str],
        gemini_api_keys: List[str],
        groq_api_keys: List[str],
        query_count: int = 0,
        max_retries_per_model: int = 2,
        gemini_cooldown: int = 60,
        groq_cooldown: int = 65,
        gemini_key_cooldown: int = 600,
        groq_key_cooldown: int = 600,
        session_state: Optional[dict] = None,
    ):
        self.gemini_models = gemini_models
        self.groq_models = groq_models
        self.gemini_api_keys = gemini_api_keys
        self.groq_api_keys = groq_api_keys
        self.query_count = query_count
        self.max_retries_per_model = max_retries_per_model
        self.cooldowns = {"gemini": gemini_cooldown, "groq": groq_cooldown}
        self.key_cooldowns = {"gemini": gemini_key_cooldown, "groq": groq_key_cooldown}

        if session_state is not None:
            session_state.setdefault("model_states", {})
            for model in gemini_models:
                if model not in session_state["model_states"]:
                    session_state["model_states"][model] = {
                        "retries": 0, "last_error_time": 0, "is_exhausted": False, "provider": "gemini"
                    }
            for model in groq_models:
                if model not in session_state["model_states"]:
                    session_state["model_states"][model] = {
                        "retries": 0, "last_error_time": 0, "is_exhausted": False, "provider": "groq"
                    }
            session_state.setdefault("key_states", {})
            for provider, keys in [("gemini", gemini_api_keys), ("groq", groq_api_keys)]:
                if provider not in session_state["key_states"]:
                    session_state["key_states"][provider] = {
                        "keys": keys, "current_index": 0,
                        "exhausted": [False] * len(keys),
                        "last_error_time": [0] * len(keys),
                        "retries": [0] * len(keys),
                    }
            self.model_states = session_state["model_states"]
            self.key_states = session_state["key_states"]
        else:
            self.model_states = {}
            for model in gemini_models:
                self.model_states[model] = {
                    "retries": 0, "last_error_time": 0, "is_exhausted": False, "provider": "gemini"
                }
            for model in groq_models:
                self.model_states[model] = {
                    "retries": 0, "last_error_time": 0, "is_exhausted": False, "provider": "groq"
                }
            self.key_states = {}
            for provider, keys in [("gemini", gemini_api_keys), ("groq", groq_api_keys)]:
                self.key_states[provider] = {
                    "keys": keys, "current_index": 0,
                    "exhausted": [False] * len(keys),
                    "last_error_time": [0] * len(keys),
                    "retries": [0] * len(keys),
                }

        self.last_provider: Optional[str] = None

    # ------------------------------------------------------------------
    # Model state helpers
    # ------------------------------------------------------------------

    def _is_model_available(self, model_name: str) -> bool:
        """"Check if a model is available (not exhausted or in cooldown)"""
        state = self.model_states.get(model_name)
        if not state:
            return False
        if state['is_exhausted']:
            cooldown = self.cooldowns.get(state['provider'], 60)
            if time.time() - state['last_error_time'] > cooldown:
                state['is_exhausted'] = False
                state['retries'] = 0
                logger.info(f"Model {model_name} cooldown expired, making available again")
                return True
            return False
        return True

    def _get_next_available_model(self, provider: str) -> Optional[str]:
        """Get the next available model for a given provider"""
        models = self.gemini_models if provider == 'gemini' else self.groq_models
        for model in models:
            if self._is_model_available(model):
                return model
        return None

    def _mark_model_exhausted(self, model_name: str, error: Exception):
        """Mark a model as exhausted after retries are exceeded"""
        state = self.model_states.get(model_name)
        if state:
            state['retries'] += 1
            state['last_error_time'] = time.time()
            if state['retries'] >= self.max_retries_per_model:
                state['is_exhausted'] = True
                logger.warning(
                    f"event=llm_model_exhausted model={model_name} provider={state['provider']} error={type(error).__name__}"
                )
            else:
                logger.info(
                    f"event=llm_retry model={model_name} provider={state['provider']} "
                    f"retry={state['retries']}/{self.max_retries_per_model}"
                )

    # ------------------------------------------------------------------
    # Key state helpers
    # ------------------------------------------------------------------

    def _is_key_available(self, provider: str, key_index: int) -> bool:
        """Check if a key is available (not exhausted or cooldown expired)"""
        state = self.key_states.get(provider)
        if not state or key_index >= len(state['keys']):
            return False
        if state['exhausted'][key_index]:
            cooldown = self.key_cooldowns.get(provider, 600)
            if time.time() - state['last_error_time'][key_index] > cooldown:
                state['exhausted'][key_index] = False
                state['retries'][key_index] = 0
                logger.info(f"Key {key_index} for {provider} cooldown expired, available again")
                return True
            return False
        return True

    def _get_active_key(self, provider: str) -> tuple:
        """Return (key_index, key_string) for the first available key, or (None, None)."""
        state = self.key_states.get(provider)
        if not state or not state['keys']:
            return None, None
        start = self.query_count % len(state['keys'])
        n = len(state['keys'])
        for offset in range(n):
            idx = (start + offset) % n
            if self._is_key_available(provider, idx):
                state['current_index'] = idx
                return idx, state['keys'][idx]
        return None, None

    def _mark_key_exhausted(self, provider: str, key_index: int):
        """Mark a specific API key as exhausted"""
        state = self.key_states.get(provider)
        if state and key_index < len(state['keys']):
            state['exhausted'][key_index] = True
            state['last_error_time'][key_index] = time.time()
            state['retries'][key_index] += 1
            logger.info(f"event=llm_key_exhausted provider={provider} key={key_index}")

    # ------------------------------------------------------------------
    # Quota / rate limit detection
    # ------------------------------------------------------------------

    def _is_quota_or_rate_error(self, error: Exception) -> bool:
        """Check if error is related to quota or rate limits"""
        if isinstance(error, (ResourceExhausted, RateLimitError)):
            return True
        if isinstance(error, APIStatusError) and getattr(error, 'status_code', None) in (429, 503):
            return True
        error_str = str(error).lower()
        quota_indicators = ['quota', 'rate_limit', 'ratelimit']
        return any(i in error_str for i in quota_indicators)

    # ------------------------------------------------------------------
    # LLM instance factory
    # ------------------------------------------------------------------

    def _create_llm_instance(self, model_name: str, provider: str, api_key: str):
        """Create an LLM instance for the given model, provider, and API key"""
        if provider == 'gemini':
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key,
            )
        elif provider == 'groq':
            return ChatGroq(
                model=model_name,
                temperature=0,
                api_key=api_key,
                timeout=45,
            )
        raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    # Invoke with fallback
    # ------------------------------------------------------------------

    def invoke_with_fallback(self, messages):
        """
        Invoke LLM with intelligent fallback, model rotation, and key rotation.

        Primary provider rotates every 2 queries.
        Within a provider, keys are tried in round-robin on quota errors.
        Only when all keys are exhausted does it mark the model exhausted.
        """
        last_error = None
        providers = ['groq', 'gemini']
        shift = (self.query_count // 2) % len(providers)
        ordered_providers = providers[shift:] + providers[:shift]

        for provider in ordered_providers:
            keys_exhausted = False
            while True:
                model_name = self._get_next_available_model(provider)
                if not model_name:
                    logger.warning(f"event=llm_no_available_models provider={provider}")
                    break

                while True:
                    key_idx, api_key = self._get_active_key(provider)
                    if key_idx is None:
                        logger.warning(f"event=llm_no_available_keys provider={provider}")
                        keys_exhausted = True
                        break

                    try:
                        logger.info(f"event=llm_attempt model={model_name} provider={provider} key={key_idx}")
                        llm_start = time.monotonic()
                        llm = self._create_llm_instance(model_name, provider, api_key)
                        result = llm.invoke(messages)
                        elapsed = time.monotonic() - llm_start
                        logger.info(
                            f"event=llm_success model={model_name} provider={provider} key={key_idx} elapsed_s={elapsed:.2f}"
                        )
                        self.last_provider = f"{provider.capitalize()} ({model_name})"
                        return result
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"event=llm_failed model={model_name} provider={provider} key={key_idx} error={type(e).__name__}"
                        )

                        if self._is_quota_or_rate_error(e):
                            self._mark_key_exhausted(provider, key_idx)
                            continue
                        elif isinstance(e, (APITimeoutError, ServiceUnavailable, GoogleAPIError)):
                            self._mark_model_exhausted(model_name, e)
                            break
                        else:
                            raise

                if keys_exhausted:
                    break

        logger.error("event=llm_all_models_exhausted")
        if last_error:
            raise LLMExhaustedError(
                f"All LLM providers exhausted. Last error: {type(last_error).__name__}: {str(last_error)}"
            )
        raise LLMExhaustedError("All LLM providers are unavailable")

    # ------------------------------------------------------------------
    # Stream with fallback
    # ------------------------------------------------------------------

    def stream_with_fallback(self, messages):
        """
        Stream LLM response with intelligent fallback, model rotation, and key rotation.
        """
        last_error = None
        providers = ['groq', 'gemini']
        shift = (self.query_count // 2) % len(providers)
        ordered_providers = providers[shift:] + providers[:shift]

        for provider in ordered_providers:
            keys_exhausted = False
            while True:
                model_name = self._get_next_available_model(provider)
                if not model_name:
                    logger.warning(f"event=llm_no_available_models provider={provider}")
                    break

                while True:
                    key_idx, api_key = self._get_active_key(provider)
                    if key_idx is None:
                        logger.warning(f"event=llm_no_available_keys provider={provider}")
                        keys_exhausted = True
                        break

                    try:
                        logger.info(f"event=llm_stream_attempt model={model_name} provider={provider} key={key_idx}")
                        llm = self._create_llm_instance(model_name, provider, api_key)
                        stream_iter = llm.stream(messages)
                        first_chunk = next(stream_iter)

                        logger.info(f"event=llm_stream_success model={model_name} provider={provider} key={key_idx}")
                        self.last_provider = f"{provider.capitalize()} ({model_name})"

                        yield first_chunk
                        try:
                            yield from stream_iter
                        except Exception as e:
                            last_error = e
                            logger.warning(
                                f"event=llm_stream_mid_failure model={model_name} provider={provider} key={key_idx} error={type(e).__name__}"
                            )
                            if self._is_quota_or_rate_error(e):
                                self._mark_key_exhausted(provider, key_idx)
                            elif isinstance(e, (APITimeoutError, ServiceUnavailable, GoogleAPIError)):
                                self._mark_model_exhausted(model_name, e)
                            raise
                        return
                    except StopIteration:
                        self.last_provider = f"{provider.capitalize()} ({model_name})"
                        return
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"event=llm_stream_failed model={model_name} provider={provider} key={key_idx} error={type(e).__name__}"
                        )

                        if self._is_quota_or_rate_error(e):
                            self._mark_key_exhausted(provider, key_idx)
                            continue
                        elif isinstance(e, (APITimeoutError, ServiceUnavailable, GoogleAPIError)):
                            self._mark_model_exhausted(model_name, e)
                            break
                        else:
                            raise

                if keys_exhausted:
                    break

        logger.error("event=llm_all_models_exhausted (streaming)")
        if last_error:
            raise LLMExhaustedError(
                f"All LLM providers exhausted. Last error: {type(last_error).__name__}: {str(last_error)}"
            )
        raise LLMExhaustedError("All LLM providers are unavailable")