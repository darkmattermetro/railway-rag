"""
Simplified LLM selector: Groq primary -> Gemini fallback, no rotation.
"""
import logging
from httpx import ConnectError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from groq import RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


class LLMExhaustedError(Exception):
    """Raised when all available LLM providers are exhausted"""
    pass


class ModelSelector:
    """
    Try Groq first; if RateLimitError/APITimeoutError/ConnectError,
    log event=groq_fallback and try Gemini.
    """
    def __init__(
        self,
        groq_api_key: str,
        gemini_api_key: str,
        groq_model: str = "llama-3.3-70b-versatile",
        gemini_model: str = "gemini-1.5-flash",
        groq_timeout: int = 45,
    ):
        self.groq_api_key = groq_api_key
        self.gemini_api_key = gemini_api_key
        self.groq_model = groq_model
        self.gemini_model = gemini_model
        self.groq_timeout = groq_timeout

    def invoke_with_fallback(self, messages):
        # 1. Groq (primary)
        logger.info("event=llm_attempt provider=groq model=%s", self.groq_model)
        try:
            llm = ChatGroq(
                model=self.groq_model,
                temperature=0,
                api_key=self.groq_api_key,
                timeout=self.groq_timeout,
            )
            result = llm.invoke(messages)
            logger.info("event=llm_groq_success model=%s", self.groq_model)
            return result
        except (RateLimitError, APITimeoutError, ConnectError) as e:
            logger.warning(
                "event=groq_fallback model=%s error=%s",
                self.groq_model,
                type(e).__name__,
            )

        # 2. Gemini (fallback)
        logger.info("event=llm_attempt provider=gemini model=%s", self.gemini_model)
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=0,
                google_api_key=api_key,
            )
            result = llm.invoke(messages)
            logger.info("event=llm_gemini_success model=%s", self.gemini_model)
            return result
        except Exception as e:
            logger.error(
                "event=llm_all_providers_exhausted error=%s",
                type(e).__name__,
            )
            raise LLMExhaustedError(
                f"All LLM providers exhausted. Last error: {type(e).__name__}: {e}"
            )
