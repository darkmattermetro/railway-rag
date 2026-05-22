"""
Model Selector for intelligent LLM fallback with rotation
"""
import time
import logging
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from groq import RateLimitError, APIStatusError, APITimeoutError
from google.api_core.exceptions import GoogleAPIError
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import streamlit as st

logger = logging.getLogger(__name__)

class LLMExhaustedError(Exception):
    """Raised when all available LLM providers are exhausted"""
    pass

class ModelSelector:
    """
    Intelligent model selector that rotates through available models
    based on quotas, rate limits, and errors.
    """
    
    def __init__(
        self,
        gemini_models: List[str],
        groq_models: List[str],
        hf_models: List[str],
        gemini_api_key: str,
        groq_api_key: str,
        hf_api_key: str,
        max_retries_per_model: int = 2,
        cooldown_seconds: int = 30
    ):
        self.gemini_models = gemini_models
        self.groq_models = groq_models
        self.hf_models = hf_models
        self.gemini_api_key = gemini_api_key
        self.groq_api_key = groq_api_key
        self.hf_api_key = hf_api_key
        self.max_retries_per_model = max_retries_per_model
        self.cooldown_seconds = cooldown_seconds
        
        # Track model states: {model_name: {'retries': int, 'last_error_time': float, 'is_exhausted': bool}}
        self.model_states = {}
        
        # Initialize model states
        for model in gemini_models:
            self.model_states[model] = {
                'retries': 0,
                'last_error_time': 0,
                'is_exhausted': False,
                'provider': 'gemini'
            }
        
        for model in groq_models:
            self.model_states[model] = {
                'retries': 0,
                'last_error_time': 0,
                'is_exhausted': False,
                'provider': 'groq'
            }
            
        for model in hf_models:
            self.model_states[model] = {
                'retries': 0,
                'last_error_time': 0,
                'is_exhausted': False,
                'provider': 'hf'
            }
        self.last_provider: Optional[str] = None
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available (not exhausted or in cooldown)"""
        state = self.model_states.get(model_name)
        if not state:
            return False
            
        # If marked as exhausted, check if cooldown has passed
        if state['is_exhausted']:
            if time.time() - state['last_error_time'] > self.cooldown_seconds:
                # Reset exhausted state after cooldown
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
        """Mark a model as exhausted due to quota/rate limit error"""
        state = self.model_states.get(model_name)
        if state:
            state['is_exhausted'] = True
            state['last_error_time'] = time.time()
            state['retries'] += 1
            logger.warning(
                f"event=llm_model_exhausted model={model_name} provider={self.model_states[model_name]['provider']} error={type(error).__name__}"
            )
    
    def _is_quota_or_rate_error(self, error: Exception) -> bool:
        """Check if error is related to quota or rate limits"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for quota/rate limit indicators
        quota_indicators = [
            'resource_exhausted',
            'quota',
            '429',
            'rate_limit',
            'ratelimit',
            'exceeded',
            'limit exceeded'
        ]
        
        # Specific exception types that indicate quota/rate issues
        quota_exception_types = [
            'ResourceExhausted',
            'RateLimitError',
            'APIStatusError'  # Will check status code below
        ]
        
        # Check exception type
        if error_type in quota_exception_types:
            if error_type == 'APIStatusError':
                # For APIStatusError, check if it's a 429 or similar
                if hasattr(error, 'code') and str(error.code) in ['429', '503']:
                    return True
                # Also check message for quota indicators
                if any(indicator in error_str for indicator in quota_indicators):
                    return True
            else:
                return True
        
        # Check error message for quota indicators
        if any(indicator in error_str for indicator in quota_indicators):
            return True
            
        return False
    
    def _create_llm_instance(self, model_name: str, provider: str):
        """Create an LLM instance for the given model and provider"""
        if provider == 'gemini':
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=self.gemini_api_key,
            )
        elif provider == 'groq':
            return ChatGroq(
                model=model_name,
                temperature=0,
                api_key=self.groq_api_key,
                timeout=45  # Default timeout
            )
        elif provider == 'hf':
            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=model_name,
                    huggingfacehub_api_token=self.hf_api_key,
                    temperature=0,
                    max_new_tokens=512
                )
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def invoke_with_fallback(self, messages):
        """
        Invoke LLM with intelligent fallback and model rotation.
        Tries models in priority order, handling quotas and rate limits.
        Logs attempts and successes in the same format as the original code.
        """
        last_error = None
        
        # Try Gemini models first
        for attempt in range(len(self.gemini_models)):
            model_name = self._get_next_available_model('gemini')
            if not model_name:
                logger.warning("event=llm_no_available_models provider=gemini")
                break
                
            try:
                logger.info(f"event=llm_attempt model={model_name} provider=gemini")
                llm_start = time.monotonic()
                llm = self._create_llm_instance(model_name, 'gemini')
                result = llm.invoke(messages)
                elapsed = time.monotonic() - llm_start
                logger.info(
                    f"event=llm_primary_success model={model_name} elapsed_s={elapsed:.2f}"
                )
                self.last_provider = "Gemini"
                return result
            except (GoogleAPIError, ResourceExhausted, ServiceUnavailable) as e:
                last_error = e
                logger.warning(
                    f"event=llm_primary_failed model={model_name} provider=gemini error={type(e).__name__}"
                )
                
                # Check if this is a quota/rate limit error that should trigger model rotation
                if self._is_quota_or_rate_error(e):
                    self._mark_model_exhausted(model_name, e)
                    logger.info(f"event=llm_quota_exceeded model={model_name} provider=gemini")
                    continue  # Try next model
                else:
                    # For non-quota Google API errors, still try next model for robustness
                    self._mark_model_exhausted(model_name, e)
                    continue
            except Exception as e:
                # Catch any other unexpected exceptions
                last_error = e
                logger.warning(
                    f"event=llm_primary_failed model={model_name} provider=gemini error={type(e).__name__}"
                )
                # Treat unexpected errors as potential quota issues to be safe
                self._mark_model_exhausted(model_name, e)
                continue
        
        # Try Groq models if Gemini models failed or unavailable
        for attempt in range(len(self.groq_models)):
            model_name = self._get_next_available_model('groq')
            if not model_name:
                logger.warning("event=llm_no_available_models provider=groq")
                break
                
            try:
                logger.info(f"event=llm_attempt model={model_name} provider=groq")
                llm_start = time.monotonic()
                llm = self._create_llm_instance(model_name, 'groq')
                result = llm.invoke(messages)
                elapsed = time.monotonic() - llm_start
                logger.info(
                    f"event=llm_fallback_success model={model_name} elapsed_s={elapsed:.2f}"
                )
                self.last_provider = "Groq"
                return result
            except (RateLimitError, APIStatusError, APITimeoutError) as e:
                last_error = e
                logger.warning(
                    f"event=llm_fallback_failed model={model_name} provider=groq error={type(e).__name__}"
                )
                
                # Check if this is a quota/rate limit error
                if self._is_quota_or_rate_error(e):
                    self._mark_model_exhausted(model_name, e)
                    logger.info(f"event=llm_quota_exceeded model={model_name} provider=groq")
                    continue  # Try next model
                else:
                    # For non-quota Groq errors, still try next model for robustness
                    self._mark_model_exhausted(model_name, e)
                    continue
            except Exception as e:
                # Catch any other unexpected exceptions
                last_error = e
                logger.warning(
                    f"event=llm_fallback_failed model={model_name} provider=groq error={type(e).__name__}"
                )
                # Treat unexpected errors as potential quota issues to be safe
                self._mark_model_exhausted(model_name, e)
                continue
        
        # Try HF models if Gemini and Groq models failed or unavailable
        for attempt in range(len(self.hf_models)):
            model_name = self._get_next_available_model('hf')
            if not model_name:
                logger.warning("event=llm_no_available_models provider=hf")
                break
                
            try:
                logger.info(f"event=llm_attempt model={model_name} provider=hf")
                llm_start = time.monotonic()
                llm = self._create_llm_instance(model_name, 'hf')
                result = llm.invoke(messages)
                elapsed = time.monotonic() - llm_start
                logger.info(
                    f"event=llm_fallback_success model={model_name} elapsed_s={elapsed:.2f}"
                )
                self.last_provider = "HF"
                return result
            except Exception as e:
                # Catch any other unexpected exceptions
                last_error = e
                logger.warning(
                    f"event=llm_fallback_failed model={model_name} provider=hf error={type(e).__name__}"
                )
                # Treat unexpected errors as potential quota issues to be safe
                self._mark_model_exhausted(model_name, e)
                continue
        
        # If we get here, all models are exhausted or failed
        logger.error("event=llm_all_models_exhausted")
        if last_error:
            raise LLMExhaustedError(
                f"All LLM providers exhausted. Last error: {type(last_error).__name__}: {str(last_error)}"
            )
        else:
            raise LLMExhaustedError("All LLM providers are unavailable")