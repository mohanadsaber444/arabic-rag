class RagError(Exception):
    """Base class for all expected application errors."""


class UnsupportedFileTypeError(RagError): ...


class FileTooLargeError(RagError): ...


class EmptyDocumentError(RagError): ...


class NeedsOcrError(RagError): ...


class EmbeddingModelMismatchError(RagError): ...


class DuplicateDocumentError(RagError): ...


class DocumentNotFoundError(RagError): ...



class ExternalServiceError(RagError): ...