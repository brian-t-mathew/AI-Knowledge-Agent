from datetime import datetime
import requests
from asteval import Interpreter
import logging
from langchain.agents import tool
import os
from serpapi.google_search import GoogleSearch  # Correct import path
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def get_current_time(input_str: str = "") -> str:
    """Get the current date and time. Use only when specifically asked for current time/date.
    Input is ignored - just pass empty string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
async def get_weather(location: str) -> str:
    """Get current weather for a specific location. Use only for weather queries."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather service unavailable - API key not configured"
        
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"Could not retrieve weather data for {location}. Error: {data.get('message', 'Unknown error')}"
        
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return (f"Weather in {location}: {weather_desc}, temperature {temp}°C, "
                f"feels like {feels_like}°C, humidity {humidity}%")

    except requests.RequestException as e:
        logger.error(f"Weather API request failed: {str(e)}")
        return f"Weather service temporarily unavailable: {str(e)}"
    except Exception as e:
        logger.error(f"Weather tool error: {str(e)}")
        return f"An error occurred while fetching weather data: {str(e)}"
    
@tool
async def search_google(query: str) -> str:
    """Search for recent news, current events, or information not in general knowledge.
    Performs a Google search using SerpAPI and parses snippets safely."""

    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return "Search service unavailable - API key not configured"

    params = {
        "q": query,
        "engine": "google",
        "api_key": api_key,
        "num": 5
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://serpapi.com/search", params=params, timeout=10)
        results = response.json()
        organic = results.get("organic_results", [])
        
        if not organic:
            return "No relevant search results found."

        snippets = [item.get("snippet", "") for item in organic if item.get("snippet")]
        snippets = [s for s in snippets if s]  # filter out empty snippets

        if not snippets:
            return "No snippets available from search results."

        return f"Top result snippet: {snippets[0]}"

    except Exception as e:
        logger.error(f"Search API error: {str(e)}")
        return f"Search temporarily unavailable due to an error: {str(e)}"

@tool
def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations. Input should be a mathematical expression like '12 * (3 + 2)'."""
    try:
        # Security: Only allow safe mathematical characters
        allowed_chars = set("0123456789+-*/(). ")
        if not all(ch in allowed_chars for ch in expression):
            return "Invalid characters in mathematical expression"
        
        # Additional security: prevent empty or suspicious expressions
        if not expression.strip() or len(expression) > 200:
            return "Invalid mathematical expression"
        
        aeval = Interpreter()
        result = aeval.eval(expression)
        
        if result is None:
            return "Could not evaluate the mathematical expression"
            
        return f"The result of {expression} is {result}"
        
    except Exception as e:
        logger.error(f"Calculator error: {str(e)}")
        return "Invalid mathematical expression - please check your syntax"