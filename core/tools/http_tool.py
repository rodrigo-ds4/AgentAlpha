"""
Generic HTTP Request Tool for external API integration
Made by Rodrigo de Sarasqueta
"""

import json
import asyncio
import httpx
from typing import Dict, Any

from .base import BaseMinimalTool


class HTTPRequestTool(BaseMinimalTool):
    """Generic tool for making HTTP requests to external APIs"""
    
    name = "http_request"
    description = """
    Make HTTP requests to external APIs to fetch data or perform operations.
    Use this when you need to get information from external services or APIs.
    
    Input should be a JSON string with request details:
    {
        "url": "https://api.example.com/endpoint",
        "method": "GET",
        "headers": {"Authorization": "Bearer token"},
        "data": {"key": "value"}
    }
    
    Supported methods: GET, POST, PUT, DELETE
    """
    
    def _run(self, request: str) -> str:
        """Synchronous HTTP request"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(request))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(request))
        except RuntimeError:
            return asyncio.run(self._arun(request))
    
    async def _arun(self, request: str) -> str:
        """Asynchronous HTTP request"""
        try:
            print(f"HTTP request: {request}")
            
            # Parse request JSON
            try:
                request_data = json.loads(request)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format. Please provide valid JSON with url, method, headers, and data fields."
            
            # Validate required fields
            if "url" not in request_data:
                return "Error: Missing 'url' field in request."
            
            url = request_data["url"]
            method = request_data.get("method", "GET").upper()
            headers = request_data.get("headers", {})
            data = request_data.get("data", {})
            
            # Configuration
            timeout = self.config.get("timeout", 30)
            # Simple HTTP request - no retries
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if method == "GET":
                        response = await client.get(url, params=data, headers=headers)
                    elif method == "POST":
                        response = await client.post(url, json=data, headers=headers)
                    elif method == "PUT":
                        response = await client.put(url, json=data, headers=headers)
                    elif method == "DELETE":
                        response = await client.delete(url, headers=headers)
                    else:
                        return f"Error: Unsupported HTTP method: {method}"
                    
                    response.raise_for_status()
                    
                    # Parse response
                    try:
                        response_data = response.json()
                        return f"HTTP Response:\n{json.dumps(response_data, indent=2)}"
                    except json.JSONDecodeError:
                        return f"HTTP Response:\n{response.text}"
                        
            except httpx.TimeoutException:
                return f"Error: Request timed out"
            except httpx.HTTPStatusError as e:
                return f"Error: HTTP {e.response.status_code}"
            except httpx.RequestError as e:
                return f"Error: Connection failed - {str(e)}"
            
        except Exception as e:
            error_msg = f"Error processing HTTP request: {str(e)}"
            print(error_msg)
            return error_msg
