"""
Custom error classes and handlers for ResearchTrader
"""

from fastapi import HTTPException


class ServiceError(HTTPException):
    """
    Custom exception for service-related errors

    Attributes:
        detail: Error message
        service: Name of the service that caused the error
        status_code: HTTP status code
    """

    def __init__(self, detail: str, service: str, status_code: int = 500, headers: dict = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.service = service
