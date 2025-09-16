"""
Base tool class for the minimal agent
Made by Rodrigo de Sarasqueta
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain.tools import BaseTool
from pydantic import Field


class BaseMinimalTool(BaseTool, ABC):
    """Base class for minimal agent tools"""
    
    config: Dict[str, Any] = Field(exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config=config, **kwargs)
    
    @abstractmethod
    def _run(self, query: str) -> str:
        """Synchronous execution"""
        pass
    
    async def _arun(self, query: str) -> str:
        """Asynchronous execution - default to sync"""
        return self._run(query)
