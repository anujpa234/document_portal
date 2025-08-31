
import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
log = CustomLogger().get_logger(__name__)


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GEMINI_API_KEY","OPENAI_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")
        # Debug: Log all available env vars that might be API keys
        debug_env = {k: v for k, v in os.environ.items() if 'API' in k or 'KEY' in k}
        log.info(f"Available env vars with API/KEY: {list(debug_env.keys())}")
        
        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys.exc_info())

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    
    """
    A utility class to load embedding models and LLM models.
    """
    
    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

        
    def _validate_env(self):
        """
        Validate necessary environment variables.
        Ensure API keys exist.
        """
        required_vars = self.config.get("required_env_vars", [])
        self.api_keys={key:os.getenv(key) for key in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise DocumentPortalException("Missing environment variables", sys)
        log.info("Environment variables validated", available_keys=[k for k in self.api_keys if self.api_keys[k]]) # "available_keys": ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
        
    def load_embeddings(self):
        """
        Load and return the embedding model.
        """
        try:
            log.info("Loading embedding model...")
            embedding_block = self.config["embedding_model"]
            provider_key = os.getenv("LLM_PROVIDER", "groq")  # Default groq
            embedding_config = embedding_block[provider_key]
            provider = embedding_config.get("provider")
            model_name = embedding_config.get("model_name")
            log.info("Loading LLM", provider=provider, model=model_name)
            
            if provider == "google":
                return GoogleGenerativeAIEmbeddings(model=model_name)
            else:
                return OpenAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)
        
    def load_llm(self):
        """
        Load and return the LLM model.
        """
        """Load LLM dynamically based on provider in config."""
        
        llm_block = self.config["llm"]

        log.info("Loading LLM...")
        
        provider_key = os.getenv("LLM_PROVIDER", "groq")  # Default groq
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        
        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            api_key = self.api_key_mgr.get("GEMINI_API_KEY")
            llm=ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            return llm

        elif provider == "groq":
            api_key = self.api_key_mgr.get("GROQ_API_KEY")
            llm=ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
            )
            return llm
            
        elif provider == "openai":
            api_key = self.api_key_mgr.get("OPENAI_API_KEY")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
    
    
if __name__ == "__main__":
    loader = ModelLoader()
    
    # Test embedding model loading
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    
    # Test LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    
    # Test the ModelLoader
    result=llm.invoke("Hello, how are you?")
    emb_result=embeddings.embed_query("Hi Everyone")
    print(f"LLM Result: {result.content}")
    print(f"Embedding Result: {emb_result}")