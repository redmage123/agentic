# services/generative_agent/domain/exceptions.py
class GenerativeServiceError(Exception):
    """Base exception for Generative Service"""
    pass

class PromptError(GenerativeServiceError):
    """Error in prompt loading or chaining"""
    pass

class LLMError(GenerativeServiceError):
    """Error in LLM interaction"""
    pass

class AnalysisError(GenerativeServiceError):
    """Error in analysis processing"""
    pass

class ValidationError(GenerativeServiceError):
    """Error in request validation"""
    pass

class CacheError(GenerativeServiceError):
    """Error in cache operations"""
    pass
