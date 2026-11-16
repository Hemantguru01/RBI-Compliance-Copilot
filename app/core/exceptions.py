"""Domain exceptions. The API layer maps these to HTTP status codes."""


class CopilotError(Exception):
    """Base class for application errors."""


class PDFExtractionError(CopilotError):
    """A PDF could not be opened or contains no extractable text."""


class IndexNotReadyError(CopilotError):
    """The vector index has not been built yet or failed to load."""


class IndexCorruptedError(CopilotError):
    """Index files exist but are inconsistent or unreadable."""


class LLMConfigurationError(CopilotError):
    """The LLM provider is not configured (e.g. missing API key)."""


class LLMGenerationError(CopilotError):
    """The LLM provider returned an error or an unusable response."""
