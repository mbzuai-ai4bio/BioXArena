#!/usr/bin/env python3
"""
Browse-Master MCP adapter
Calls existing FastAPI service
"""

import os
import json
import asyncio
from mcp.server.fastmcp import FastMCP
import aiohttp

# Configuration
MCP_PORT = int(os.getenv("MCP_PORT", "8002"))
HOST = os.getenv("HOST", "0.0.0.0")

# Your FastAPI service address
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:1234")

# Read API key from config file or environment variable
def _load_serper_api_key():
    """Read Serper API key from environment variable or config file"""
    key = os.getenv("SERPER_API_KEY")
    if key:
        return key
    # Try to read from config file
    config_path = os.path.join(os.path.dirname(__file__), "../configs/web_agent.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("serper_api_key", "")
    return ""

SERPER_API_KEY = _load_serper_api_key()

# initialize MCP server
mcp = FastMCP(
    "browse-master-search-tools",
    host=HOST,
    port=MCP_PORT,
)

async def make_async_request(session, url, payload, timeout=30):
    """async HTTP request"""
    async with session.post(url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()

@mcp.tool()
async def web_search(query: str, top_k: int = 10) -> str:
    """
    Search web by Google
    
    Args:
        query: Search keyword
        top_k: the number of outcomes, default is 10
    """
    try:
        payload = {
            "query": query,
            "serper_api_key": SERPER_API_KEY,
            "top_k": top_k
        }
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session, 
                f"{API_BASE_URL}/search", 
                payload
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP request error: {str(e)}"
    except asyncio.TimeoutError:
        return "Request timeout, please try again later"
    except Exception as e:
        return f"Search error: {str(e)}"

@mcp.tool()
async def web_parse(link: str, user_prompt: str, llm: str = "gpt-4o") -> str:
    """
    Parse and analyze web page content
    
    Args:
        link: URL
        user_prompt: Specific query or analysis request about web page content
        llm: llm model, default is gpt-4o
    """
    try:
        payload = {
            "link": link,
            "user_prompt": user_prompt,
            "llm": llm
        }
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/web_parse",
                payload,
                timeout=60
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP request error: {str(e)}"
    except asyncio.TimeoutError:
        return "Request timeout, please check your URL"
    except Exception as e:
        return f"Error parsing web page content: {str(e)}"

@mcp.tool()
async def batch_search_and_filter(keyword: str) -> str:
    """
    Batch search and filter results
    
    Args:
        keyword: Search keyword
    """
    try:
        payload = {
            "keyword": keyword
        }
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/batch_search_and_filter",
                payload,
                timeout=120
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP request error: {str(e)}"
    except asyncio.TimeoutError:
        return "Request timeout, search may take longer"
    except Exception as e:
        return f"Search error: {str(e)}"

@mcp.tool()
async def generate_keywords(seed_keyword: str) -> str:
    """
    Generate multiple search keywords
    
    Args:
        seed_keyword: Seed keyword
    """
    try:
        payload = {
            "seed_keyword": seed_keyword
        }
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/generate_keywords",
                payload
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP request error: {str(e)}"
    except asyncio.TimeoutError:
        return "Request timeout"
    except Exception as e:
        return f"Error generating keywords: {str(e)}"

@mcp.tool()
async def check_condition(content: str, condition: str) -> str:
    """
    Evaluate whether content meets specified conditions
    
    Args:
        content: Content to evaluate
        condition: Evaluation condition
    """
    try:
        payload = {
            "content": content,
            "condition": condition
        }
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/check_condition",
                payload
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP request error: {str(e)}"
    except asyncio.TimeoutError:
        return "Request timeout"
    except Exception as e:
        return f"Error checking conditions: {str(e)}"

@mcp.tool()
async def pdf_read(url: str) -> str:
    """
    Read PDF from web
    
    Args:
        url: the adress of the web
    """
    try:
        payload = {"url": url}
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/read_pdf",
                payload,
                timeout=60
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP request error: {str(e)}"
    except asyncio.TimeoutError:
        return "Request timeout, the PDF is too large"
    except Exception as e:
        return f"Reading PDF error: {str(e)}"

if __name__ == "__main__":
    # run MCP server
    mcp.run(transport="streamable-http")