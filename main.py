from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import requests
import uuid
import pandas as pd
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv
import os
import re
import httpx
import asyncio
import ast
from functools import wraps
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReplaceOne
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Merger Search API",
    description="API for identifying and analyzing retail consulting firms for potential mergers, using Lyzr agent API with structured JSON responses. Configuration is managed via environment variables in a .env file.",
    version="1.0.0"
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async_client = httpx.AsyncClient(timeout=90.0)

# Load Lyzr agent API settings from environment variables
LYZR_AGENT_URL = os.getenv("LYZR_AGENT_URL")
LYZR_API_KEY = os.getenv("LYZR_API_KEY")
LYZR_USER_ID = os.getenv("LYZR_USER_ID")
LYZR_PERPLEXITY_SESSION_ID = os.getenv("LYZR_PERPLEXITY_SESSION_ID")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")

# Validate required environment variables
required_env_vars = {
    "LYZR_AGENT_URL": LYZR_AGENT_URL,
    "LYZR_API_KEY": LYZR_API_KEY,
    "LYZR_USER_ID": LYZR_USER_ID,
    "LYZR_PERPLEXITY_SESSION_ID": LYZR_PERPLEXITY_SESSION_ID,
    "APOLLO_API_KEY": APOLLO_API_KEY
}
for var_name, var_value in required_env_vars.items():
    if not var_value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise ValueError(f"Environment variable {var_name} is not set")

# Lyzr agent API headers
LYZR_AGENT_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": LYZR_API_KEY
}

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "merger_search")

# Initialize MongoDB client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]


# Collections
prompt_results_collection = db["prompt_results"]
merger_search_collection = db["merger_search"]
companies_collection = db["companies"]


@app.on_event("startup")
async def startup_db_check(): 
    try:
        await mongo_client.admin.command('ping')
        print("✅ Connected to MongoDB")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)


def sanitize_json_string(json_str: any) -> str:
    """
    Comprehensive JSON sanitization with multi-stage repair capabilities
    Handles smart quotes, unescaped characters, and complex code blocks
    """
    json_str = str(json_str).strip()
    if not json_str:
        return "{}"

    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', json_str, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()

    json_str = re.sub(r'[\x00-\x09\x0b-\x1f\x7f]', '', json_str)
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '\u2028': ' ', '\u2029': ' ', '\ufeff': ''
    }
    for bad, good in replacements.items():
        json_str = json_str.replace(bad, good)

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        error_pos = e.pos
        logger.warning(f"Initial parse failed at {error_pos}: {e}")

    try:
        if "Unterminated string" in str(e):
            quote_char = json_str[error_pos-1]
            end_quote = json_str.find(quote_char, error_pos)
            if end_quote == -1:
                json_str += quote_char

        stack = []
        for i, c in enumerate(json_str):
            if c in '{[':
                stack.append(c)
            elif c in '}]':
                if not stack:
                    json_str = json_str[:i] + json_str[i+1:]
                    continue
                last = stack.pop()
                if (c == '}' and last != '{') or (c == ']' and last != '['):
                    json_str = json_str[:i] + json_str[i+1:]

        while stack:
            json_str += '}' if stack.pop() == '{' else ']'

        return json.dumps(json.loads(json_str), indent=None)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(json_str)
        return json.dumps(parsed)
    except Exception as e:
        logger.error(f"Literal eval fallback failed: {e}")

    if json_str.startswith('{'):
        return f'[{json_str}]'
    return json_str

# Retry decorator for API calls
def async_retry(max_attempts: int = 3, delay: float = 2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    # Check if result is a dictionary with expected keys for enrich_company
                    if func.__name__ == "enrich_company":
                        if isinstance(result, dict) and ("name" in result or "error" in result):
                            return result
                        logger.warning(f"Invalid enrich_company result on attempt {attempt + 1}: {result}")
                    # Original check for Lyzr API responses
                    elif result and isinstance(result.get("response"), str):
                        try:
                            json.loads(sanitize_json_string(result["response"]))
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON on attempt {attempt + 1}")
                    logger.warning(f"API call returned None or invalid response on attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
            logger.error(f"API call failed after {max_attempts} attempts")
            return None
        return wrapper
    return decorator

# Search prompts with agent-ID
search_prompts = [
    {
        "title": "Initial Target Identification",
        "agent-ID": "6825f6b2c8abf8bac32c4d75",
        "content": """Identify boutique retail consulting firms for merger potential with the following criteria:

Target Firms:
- Headquartered in North America, Europe
- Annual revenue: $10-500M (exclude firms with revenue outside this range)
- Specializations: Product Development, Sourcing Strategy, Supply Chain Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 15-30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, industry reports (e.g., Retail Dive, Consulting.us), and LinkedIn
- Validate through public records or client testimonials; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-500M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate
- Operational Scale: Employee count
- Client Portfolio: Key retail clients
- Leadership: Key executives
- Merger Fit: Synergies

Output Requirements:
- Return a JSON array of 15-30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $10-500M revenue criterion""",
        "sources": [
            "https://www.consulting.us/rankings/top-consulting-firms-in-the-us-by-industry-expertise/retail",
            "https://managementconsulted.com/boutique-consulting-firms-in-us/",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Market Analysis and Competitive Landscape",
        "agent-ID": "6825f7cf596f910103311b2f",
        "content": """Analyze the competitive landscape of boutique retail consulting firms for merger potential:

Target Firms:
- Headquartered in North America, Europe
- Annual revenue: $10-500M (exclude firms with revenue outside this range)
- Specializations: Product Development, Sourcing Strategy, Procurement Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 15-30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from industry reports (e.g., IBISWorld), company websites, and LinkedIn
- Validate through client testimonials or public records; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-500M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate
- Operational Scale: Employee count
- Client Portfolio: Key retail clients
- Leadership: Key executives
- Merger Fit: Synergies

Output Requirements:
- Return a JSON array of 15-30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $10-500M revenue criterion""",
        "sources": [
            "https://www.ibisworld.com/",
            "https://www.linkedin.com/",
            "https://www.retaildive.com/"
        ]
    },
    {
        "title": "Sourcing and Procurement Specialists",
        "agent-ID": "6825f789c8abf8bac32c4d83",
        "content": """Conduct a survey of boutique procurement strategy consulting firms for merger potential:

Target Firms:
- Headquartered in North America, Europe
- Annual revenue: $10-500M (exclude firms with revenue outside this range)
- Specializations: Strategic Sourcing, Global Procurement, Retail Supply Chain Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 15-30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, industry reports (e.g., Procurement Leaders), and LinkedIn
- Validate through public records or client testimonials; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-500M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate
- Operational Scale: Employee count
- Client Portfolio: Key retail clients
- Leadership: Key executives
- Merger Fit: Synergies

Output Requirements:
- Return a JSON array of 15-30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $10-500M revenue criterion""",
        "sources": [
            "https://www.procurementleaders.com/industry-insight/consulting-rankings",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Product Development Expertise",
        "agent-ID": "6825f75573107c158d6094de",
        "content": """Identify boutique retail consulting firms with expertise in product development for merger potential:

Target Firms:
- Headquartered in North America, Europe
- Annual revenue: $10-500M (exclude firms with revenue outside this range)
- Specializations: Concept Development, Product Optimization, Launch Strategy
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 15-30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, industry reports (e.g., Retail Dive), and LinkedIn
- Validate through public records or client testimonials; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-500M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate
- Operational Scale: Employee count
- Client Portfolio: Key retail clients
- Leadership: Key executives
- Merger Fit: Synergies

Output Requirements:
- Return a JSON array of 15-30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $10-500M revenue criterion""",
        "sources": [
            "https://www.retaildive.com/",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Supply Chain Specialists",
        "agent-ID": "6825f6e9c8abf8bac32c4d76",
        "content": """Identify boutique retail consulting firms with expertise in supply chain optimization for merger potential:

Target Firms:
- Headquartered in North America, Europe
- Annual revenue: $10-500M (exclude firms with revenue outside this range)
- Specializations: Supply Chain Optimization, Vendor Assessment/Evaluation
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 15-30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, industry reports (e.g., Supply Chain Dive), and LinkedIn
- Validate through public records or client testimonials; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-500M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate
- Operational Scale: Employee count
- Client Portfolio: Key retail clients
- Leadership: Key executives
- Merger Fit: Synergies

Output Requirements:
- Return a JSON array of 15-30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $10-500M revenue criterion""",
        "sources": [
            "https://www.supplychaindive.com/",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Acquisition Target Identification - Digital Manufacturing",
        "agent-ID": "682f0bed792e81c46d3d6908",
        "content": """Identify boutique digital manufacturing consulting or solution firms for acquisition potential:

Target Firms:
- Headquartered in Japan or Germany
- Annual revenue: $5M–$25M (ideal: ~$20M; exclude firms with revenue outside this range)
- Ownership: Startups, privately held companies, or young firms (exclude large public companies such as Siemens, Hitachi, SAP)
- Industry focus: Oil & Gas, Energy, Mining, Utilities
- Client base: Medium to large enterprises
- Specializations: Digital Manufacturing, Industrial IoT, Asset Performance Management, Digital Plant solutions, Predictive Analytics, Manufacturing Excellence

Research Methodology:
- Identify 15–30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, startup databases (e.g., Tracxn, PitchBook), and industry publications
- Validate through public records, client testimonials, or investor databases; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $5M–$25M
- Look for companies using terminology such as:
  - "Digital Plant solutions"
  - "Industrial IoT"
  - "Predictive Asset Analytics"
  - "Asset Performance Management"
  - "Intelligent Asset Management Strategy"
  - "Manufacturing Excellence"
  - "Digital Factory Applications"

Evaluation Criteria:
- Financial Performance: Estimated revenue, recent growth trajectory
- Operational Scale: Year founded, employee count
- Client Portfolio: Major clients in industrial sectors
- Leadership: Key executives
- Merger Fit: Alignment with reference company Boslan

Output Requirements:
- Return a JSON array of 15–30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $5M–$25M revenue criterion""",
        "sources": [
            "https://www.industryweek.com/technology-and-iiot/digital-tools",
            "https://www.isa.org/intech-home/digital-transformation",
            "https://www.processingmagazine.com/digital-transformation",
            "https://pitchbook.com/profiles/company/acquisition-targets",
            "https://tracxn.com/explore/Digital-Manufacturing-Startups-in-Japan",
            "https://tracxn.com/explore/Digital-Manufacturing-Startups-in-Germany"
        ]
    },
    {
        "title": "Japanese Digital Manufacturing Acquisition Targets",
        "agent-ID": "682f0c55ced2bfaff52a8b10",
        "content": """Identify boutique digital manufacturing firms in Japan for acquisition potential:

Target Firms:
- Headquartered in Japan (international operations acceptable)
- Annual revenue: $15M–$25M (ideal: ~$20M; exclude firms outside this range)
- Ownership: Privately held, not subsidiaries of larger corporations
- Age: Established companies (in business for 5+ years); exclude early-stage startups
- Industry focus: Digital manufacturing solutions for Oil & Gas, Energy, Mining, Utilities
- Client base: Primarily enterprise industrial clients
- Specializations must include at least one of the following:
  - Digital Plant Solutions
  - Industrial IoT Implementations
  - Asset Performance Management
  - Predictive Maintenance Systems
  - Digital Twin Technology
  - Manufacturing Execution Systems

Research Methodology:
- Identify 15–30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, investor databases (e.g., PitchBook, Tracxn), and industry publications
- Validate through public records or client testimonials; data must be confirmed by at least two credible sources listed below
- Exclude large corporations, publicly traded companies, early-stage startups, or firms with unverifiable revenue

Evaluation Criteria:
- Financial Performance: Estimated revenue and growth trajectory
- Operational Scale: Year founded, employee count (if available)
- Client Portfolio: Enterprise clients in industrial sectors
- Leadership: Key executives (if available)
- Merger Fit: Synergies, competitiveness, and differentiation

Output Requirements:
- Return a JSON array of 15–30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $15M–$25M revenue criterion""",
        "sources": [
            "https://tracxn.com/explore/Digital-Manufacturing-Startups-in-Japan",
            "https://www.jpx.co.jp/english/markets/",
            "https://www.industryweek.com/technology-and-iiot/article/companies-in-japan",
            "https://www.automationworld.com/factory/iiot/article/japanese-industrial-iot",
            "https://pitchbook.com/profiles/investor/japan-industrial-companies"
        ]
    },
    {
        "title": "German & EU-Based Digital Manufacturing Acquisition Targets",
        "agent-ID": "682f0cf3792e81c46d3d6915",
        "content": """Identify boutique digital manufacturing firms in Germany for acquisition potential:

Target Firms:
- Headquartered in Germany (international operations acceptable)
- Annual revenue: $15M–$25M (ideal: ~$20M; exclude firms outside this range)
- Ownership: Privately held, not subsidiaries of larger corporations
- Age: Established companies (in business for 5+ years); exclude early-stage startups
- Industry focus: Digital manufacturing solutions for Oil & Gas, Energy, Mining, Utilities
- Client base: Primarily enterprise industrial clients
- Specializations must include at least one of the following:
  - Digital Plant Solutions
  - Industrial IoT Implementations
  - Asset Performance Management
  - Predictive Maintenance Systems
  - Digital Twin Technology
  - Manufacturing Execution Systems

Research Methodology:
- Identify 15–30 firms meeting all criteria [NOT LESS THAN 15, THIS IS IMPORTANT]
- Cross-reference data from company websites, investor databases (e.g., PitchBook, Tracxn), and industry publications
- Validate through public records or client testimonials; data must be confirmed by at least two credible sources listed below
- Exclude large corporations, publicly traded companies, early-stage startups, or firms with unverifiable revenue

Evaluation Criteria:
- Financial Performance: Estimated revenue and growth trajectory
- Operational Scale: Year founded, employee count (if available)
- Client Portfolio: Enterprise clients in industrial sectors
- Leadership: Key executives (if available)
- Merger Fit: Synergies, competitiveness, and differentiation

Output Requirements:
- Return a JSON array of 15–30 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "key_clients": ["<string>"],
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>"
        }
      ],
      "merger_synergies": "<string>",
      "Industries": "<string>",
      "Services": "<string>",
      "Broad Category": "<string>",
      "Ownership": "<string> e.g, Private, JV, Partnership, Public",
      "sources": ["<string>"]
    }
  ]
  ```
- Include 2 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, exclude from results
- Return JSON only, with no narrative text or comments
- Ensure all firms meet the $15M–$25M revenue criterion""",
        "sources": [
            "https://tracxn.com/explore/Digital-Manufacturing-Startups-in-Germany",
            "https://www.vdma.org/en/",
            "https://www.industryweek.com/technology-and-iiot/article/companies-in-germany",
            "https://www.automationworld.com/factory/iiot/article/german-industrial-iot",
            "https://pitchbook.com/profiles/investor/german-industrial-companies"
        ]
    },
    {
        "title": "Asset Management Excellence Acquisition Targets",
        "agent-ID": "682f0de8ced2bfaff52a8b34",
        "content": """I'm seeking mid-sized companies in Japan or Germany specializing in industrial asset management that would be suitable acquisition targets. Please provide detailed information on companies meeting these EXACT criteria:

- Revenue: Approximately $20M annual revenue (range $15M–$25M)
- Core business: Asset performance management, reliability solutions for industrial clients
- Target sectors: Oil & Gas, Energy, Mining, Utilities
- Status: Privately-held companies that could be acquired (NOT subsidiaries or public firms)
- Growth profile: Established businesses with recent growth and operational maturity

Focus exclusively on companies using terminology such as:
- "Reliability/Asset Performance Management"
- "Asset Reliability Consulting"
- "Intelligent Asset Management Strategy"
- "EAM/Maintenance and Reliability"
- "Predictive Asset Analytics"

Each company returned **must** include the following fields in this exact response format:
```json
[
  {
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "key_clients": ["<string>"],
    "leadership": [
      {
        "name": "<string>",
        "title": "<string>"
      }
    ],
    "merger_synergies": "<string>",
    "Industries": "<string>",
    "Services": "<string>",
    "Broad Category": "<string>",
    "Ownership": "<string> e.g, Private, JV, Partnership, Public",
    "sources": ["<string>"]
  }
]
```

Additional rules:
- Include only real, verifiable companies in the $15M–$25M revenue range
- Include a minimum of 15 and maximum of 30 companies
- Use at least two reliable public sources to validate each record
- Exclude large enterprises, public companies, and early-stage startups
- Do not include companies without confirmed revenue estimates or unclear ownership

Make sure all companies focus on asset management for industrial sectors (Oil & Gas, Energy, Mining, Utilities).""",
        "sources": [
            "https://www.reliabilityweb.com/directory",
            "https://www.plantservices.com/vendors/products/asset-management",
            "https://www.upkeep.com/learning/asset-management",
            "https://pitchbook.com/profiles/industry/asset-management-companies",
            "https://tracxn.com/explore/Asset-Management-Startups-in-Japan",
            "https://tracxn.com/explore/Asset-Management-Startups-in-Germany"
        ]
    },
#     {
#         "title": "Industrial IoT Acquisition Targets",
#         "agent-ID": "682f0e62490dde168e7cdc9b",
#         "content": """I need to identify acquisition targets specializing in Industrial IoT solutions in Japan or Germany. Please provide detailed information on companies meeting these EXACT criteria:

# - Revenue: Approximately $20M annual revenue (range $15M–$25M)
# - Core focus: Industrial IoT solutions for asset-intensive industries
# - Target markets: Oil & Gas, Energy, Mining, Utilities sectors
# - Status: Privately-held companies suitable for acquisition (NOT subsidiaries or public firms)
# - Stage: Growth-stage companies with proven market traction

# Specifically focus on companies using terms such as:
# - "Industrial IoT"
# - "IIoT/Edge Consulting"
# - "Digital Factory Applications"
# - "IoT-enabled asset monitoring"
# - "Sensor networks for industrial environments"
# - "Edge computing for operational technology"

# Each company must be presented using the following structured JSON response format:
# ```json
# [
#   {
#     "name": "<string>",
#     "domain_name": "<string>",
#     "estimated_revenue": "<string, e.g., $15M>",
#     "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
#     "employee_count": "<string, e.g., 50-75 employees>",
#     "key_clients": ["<string>"],
#     "leadership": [
#       {
#         "name": "<string>",
#         "title": "<string>"
#       }
#     ],
#     "merger_synergies": "<string>",
#     "Industries": "<string>",
#     "Services": "<string>",
#     "Broad Category": "<string>",
#     "Ownership": "<string> e.g, Private, JV, Partnership, Public",
#     "sources": ["<string>"]
#   }
# ]
# ```

# Strict criteria:
# - Include only real, verifiable companies with $15M–$25M in revenue
# - Exclude public companies, large corporations, or early-stage startups
# - Use at least two reliable public sources to validate each company
# - All companies must operate in or serve industrial sectors in Japan or Germany

# Do not include companies that are primarily software or IT firms unless their core focus is Industrial IoT for asset-intensive operations.""",
#         "sources": [
#             "https://iot-analytics.com/top-10-industrial-iot-companies/",
#             "https://tracxn.com/explore/Industrial-IoT-Startups-in-Japan",
#             "https://tracxn.com/explore/Industrial-IoT-Startups-in-Germany",
#             "https://www.iothub.de/companies/",
#             "https://pitchbook.com/profiles/industry/iot-companies-japan-germany"
#         ]
#     },
#     {
#         "title": "Supply Chain Excellence & Digital Plant Acquisition Targets",
#         "agent-ID": "682f0ed6ced2bfaff52a8b40",
#         "content": """I'm seeking mid-sized companies in Japan or Germany that specialize in Supply Chain Excellence and Digital Plant solutions for industrial clients. Please identify potential acquisition targets meeting these EXACT criteria:

# - Revenue: Approximately $20M annual revenue (range $15M–$25M)
# - Core focus: Supply chain optimization and digital plant solutions
# - Target industries: Oil & Gas, Energy, Mining, Utilities
# - Status: Privately-held companies suitable for acquisition (NOT public or subsidiaries)
# - Growth profile: Established companies with proven solutions and market traction

# Focus on companies that use specific terms such as:
# - "Supply Chain Excellence"
# - "Digital Plant solutions"
# - "Digital supply chain transformation"
# - "Integrated supply chain visibility"
# - "Plant digitalization platforms"
# - "Digital twin technology for industrial operations"

# Each company must be presented using the following structured JSON response format:
# ```json
# [
#   {
#     "name": "<string>",
#     "domain_name": "<string>",
#     "estimated_revenue": "<string, e.g., $15M>",
#     "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
#     "employee_count": "<string, e.g., 50-75 employees>",
#     "key_clients": ["<string>"],
#     "leadership": [
#       {
#         "name": "<string>",
#         "title": "<string>"
#       }
#     ],
#     "merger_synergies": "<string>",
#     "Industries": "<string>",
#     "Services": "<string>",
#     "Broad Category": "<string>",
#     "Ownership": "<string> e.g, Private, JV, Partnership, Public",
#     "sources": ["<string>"]
#   }
# ]
# ```

# Strict filtering rules:
# - Only include real, verifiable companies with $15M–$25M in revenue
# - Exclude public companies, large corporations, or very early-stage startups
# - Use reliable public sources to validate each company and their capabilities
# - All companies must serve industrial sectors in Japan or Germany

# Emphasize companies with strong digital technology portfolios that enhance industrial supply chain performance and plant operations.""",
#         "sources": [
#             "https://www.supplychaindigital.com/top10/top-10-supply-chain-companies",
#             "https://tracxn.com/explore/Supply-Chain-Management-Startups-in-Japan",
#             "https://tracxn.com/explore/Supply-Chain-Management-Startups-in-Germany",
#             "https://www.gartner.com/en/supply-chain/insights",
#             "https://pitchbook.com/profiles/industry/supply-chain-companies"
#         ]
#     }
]


# Pydantic models for request/response validation
class PromptRequest(BaseModel):
    prompt_index: int
    custom_message: Optional[str] = None

class CompaniesResponse(BaseModel):
    companies: List[str]

class AnalysisRequest(BaseModel):
    companies: List[str]
    raw_responses: Dict[str, str]

class MergerSearchResponse(BaseModel):
    results: Dict
    message: str

class EnrichCompanyRequest(BaseModel):
    company_domain: str

class FetchCompanyPerplexityRequest(BaseModel):
    company_name: str
    company_domain: str

class CompanyData(BaseModel):
    name: str
    domain_name: str
    estimated_revenue: str
    revenue_growth: str
    employee_count: str
    key_clients: List[str]
    leadership: List[Dict[str, str]]
    merger_synergies: str
    Industries: str
    Services: str
    Broad_Category: str
    Ownership: str
    sources: List[str]
    office_locations: Optional[List[str]] = None
    validation_warnings: Optional[List[str]] = None

class AnalysisRequest(BaseModel):
    companies: List[CompanyData]

# Apollo API enrichment function
@async_retry(max_attempts=3, delay=2.0)
async def enrich_company(company_domain: str) -> Dict:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.apollo.io/api/v1/organizations/enrich",
                params={"domain": company_domain},
                headers={"x-api-key": APOLLO_API_KEY}
            )
            response.raise_for_status()
            data = response.json()
            org = data.get("organization", {})
            if not org:
                logger.warning(f"No organization data returned for domain {company_domain}")
                return {"error": "No organization data found"}
            return org  # Return the full organization object
        except Exception as e:
            logger.error(f"Apollo API error for domain {company_domain}: {str(e)}")
            return {"error": str(e)}
        

@async_retry(max_attempts=3, delay=2.0)
async def query_lyzr_agent(agent_id: str, session_id: str, prompt: str) -> Optional[Dict]:
    print("payload", prompt)
    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": agent_id,
        "session_id": session_id,
        "message": prompt
    }

    print("PAYLOAD FOR THE CALL", payload)
    try:
        response = await async_client.post(
            LYZR_AGENT_URL,
            json=payload,
            headers=LYZR_AGENT_HEADERS
        )
        response.raise_for_status()
        json_response = response.json()
        raw_response = json_response.get("response", "{}")
        logger.info(f"API response size: {len(str(json_response))} bytes")
        json_response["response"] = sanitize_json_string(raw_response)
        print("RESPONSE", json_response)
        return json_response
    except Exception as e:
        logger.error(f"Lyzr agent API error: {e}")
        raise
    
async def query_perplexity(prompt: str, agent_id: str) -> Optional[Dict]:
    return await query_lyzr_agent(agent_id, LYZR_PERPLEXITY_SESSION_ID, prompt)

async def process_prompt_batch(prompt_data: Dict) -> Dict:
    try:
        response = await query_perplexity(prompt_data['content'], prompt_data['agent-ID'])
        if not response:
            return {"error": "API request failed", "title": prompt_data['title']}

        processed = await process_api_response(response, prompt_data)
        return {
            "title": prompt_data['title'],
            "response": processed.get("response", []),
            "companies": processed.get("companies", []),
            "validation_warnings": processed.get("validation_warnings", [])
        }
    except Exception as e:
        logger.error(f"Prompt processing failed for {prompt_data['title']}: {str(e)}")
        return {
            "title": prompt_data['title'],
            "response": [],
            "companies": [],
            "validation_warnings": [f"Processing error: {str(e)}"]
        }

def extract_companies(json_content: any) -> List[str]:
    companies = []
    def recursive_search(node):
        if isinstance(node, dict):
            if 'name' in node and isinstance(node['name'], str):
                companies.append(node['name'])
            for value in node.values():
                recursive_search(value)
        elif isinstance(node, list):
            for item in node:
                recursive_search(item)
    try:
        recursive_search(json_content)
        return sorted(list(set(filter(None, companies))))
    except Exception as e:
        logger.error(f"Deep extraction failed: {e}")
        return []
    

async def analyze_companies_with_claude(companies: List[Dict]) -> Optional[Dict]:
    company_data_str = json.dumps(companies, indent=2)
    payload = { 
        "user_id": "pranav@lyzr.ai",
        "agent_id": "68300c29ced2bfaff52a9215",
        "session_id": "68300c29ced2bfaff52a9215-xoemkzyvltm",
        "message": company_data_str
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LYZR_API_KEY
    }
    try:
        response = await async_client.post(
            "https://agent-prod.studio.lyzr.ai/v3/inference/chat/",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        json_response = response.json()
        raw_content = json_response.get("response", "{}")
        logger.debug(f"Raw Claude API response:\n{raw_content[:2000]}")
        Path("api_logs").mkdir(exist_ok=True)
        log_file = f"api_logs/claude_analysis_{time.time()}.json"
        with open(log_file, "w") as f:
            f.write(raw_content)
        sanitized = sanitize_json_string(raw_content)
        json_content = json.loads(sanitized)
        analysis = json_content.get("analysis", {})
        if not analysis:
            logger.error("Claude response missing 'analysis' key")
            return {}
        return analysis
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Claude API request failed: {e}")
        return {}


async def analyze_with_claude(companies: List[str], raw_responses: Dict[str, str]) -> Optional[Dict]:
    full_context = "\n\n".join([f"### {title}\n{content}" for title, content in raw_responses.items()])
    company_list = "\n".join([f"- {company}" for company in companies])
    analysis_prompt = """
Analyze the provided list of boutique retail consulting firms for merger potential based on the following criteria weightings:
- Financial Health (30%): Revenue, growth rate, profitability, valuation multiples
- Strategic Fit (30%): Service overlap, client base compatibility, market positioning
- Operational Compatibility (20%): Geographic presence, employee count, technology stack
- Leadership and Innovation (10%): Leadership expertise, proprietary methodologies, innovation metrics
- Cultural and Integration Risks (10%): Cultural alignment, integration challenges

Input Data:
- Companies: {company_list}
- Detailed Data: {full_context}

Output Requirements:
- Return a JSON object with the following structure:
  ```json
  {
    "analysis": {
      "rankings": [
        {
          "name": "<string>",
          "overall_score": <float, 0-100>,
          "financial_health_score": <float, 0-100>,
          "strategic_fit_score": <float, 0-100>,
          "operational_compatibility_score": <float, 0-100>,
          "leadership_innovation_score": <float, 0-100>,
          "cultural_integration_score": <float, 0-100>,
          "rationale": "<string>"
        }
      ],
      "recommendations": [
        {
          "name": "<string>",
          "merger_potential": "<string>",
          "key_synergies": ["<string>"],
          "potential_risks": ["<string>"]
        }
      ],
      "summary": "<string>"
    }
  }
  ```
- Include only the companies from the provided company_list
- Return JSON only, with no narrative text
"""
    response = await query_perplexity(analysis_prompt.format(company_list=company_list, full_context=full_context), "6817b49c2f31429a00a39d39")  # Using first agent-ID for Claude
    if response:
        try:
            json_content = json.loads(sanitize_json_string(response.get("response", "{}")))
            return json_content.get("analysis", {})
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            return {}
    return {}

async def process_api_response(response: Dict, prompt_data: Dict) -> Dict:
    raw_content = response.get('response', '')
    logger.debug(f"Raw Perplexity API response:\n{raw_content[:2000]}")
    Path("api_logs").mkdir(exist_ok=True)
    log_file = f"api_logs/{time.time()}.json"
    with open(log_file, "w") as f:
        f.write(raw_content)
    
    sanitized = sanitize_json_string(raw_content)
    try:
        json_content = json.loads(sanitized)
    except json.JSONDecodeError as e:
        logger.error(f"Critical JSON failure: {e}")
        return {
            "error": "Invalid JSON structure",
            "diagnosis": diagnose_json_issues(sanitized),
            "raw_sample": raw_content[:500],
            "title": prompt_data["title"],
            "sources": prompt_data["sources"],
            "companies": [],
            "validation_warnings": ["Invalid JSON response"]
        }
    
    validated_companies = []
    validation_warnings = []
    for company in json_content if isinstance(json_content, list) else []:
        if not isinstance(company, dict):
            continue
        required_fields = ["name", "domain_name", "estimated_revenue", "employee_count"]
        missing_fields = [field for field in required_fields if not company.get(field)]
        if missing_fields:
            validation_warnings.append(f"Company {company.get('name', 'unknown')} missing fields: {missing_fields}")
            continue
        revenue_str = company.get("estimated_revenue", "")
        

        logger.info(f"Fetching Apollo data for {company['name']} with domain {company['domain_name']}")
        await asyncio.sleep(0.5)
        apollo_data = await enrich_company(company.get("domain_name"))
        if "error" not in apollo_data:
            validated_company = validate_company_data(company, apollo_data)
            validated_companies.append(validated_company)
            validation_warnings.extend(validated_company.get("validation_warnings", []))
        else:
            validated_company = company.copy()
            validated_company["validation_warnings"] = [f"Apollo API failed: {apollo_data['error']}"]
            validated_companies.append(validated_company)
            validation_warnings.append(f"Apollo API failed for {company['name']}: {apollo_data['error']}")

    companies = [company["name"] for company in validated_companies]
    return {
        "title": prompt_data["title"],
        "raw_response": validated_companies,
        "companies": companies,
        "sources": prompt_data["sources"],
        "validation_warnings": validation_warnings
    }

def validate_company_data(perplexity_data: Dict, apollo_data: Dict) -> Dict:
    validated = perplexity_data.copy()
    validation_warnings = []

    if apollo_data.get("estimated_num_employees"):
        apollo_employees = apollo_data["estimated_num_employees"]
        try:
            apollo_count = int(apollo_employees)
            validated["employee_count"] = f"{apollo_count} employees"
        except (ValueError, TypeError):
            validation_warnings.append(f"Invalid Apollo employee count: {apollo_employees}")
    else:
        validation_warnings.append("Apollo data missing employee count")

    city = apollo_data.get("city", "")
    state = apollo_data.get("state", "")
    country = apollo_data.get("country", "")
    if city and state and country:
        validated["office_locations"] = [f"{city}, {state}, {country}"]
    elif city or state or country:
        # Include partial location if some fields are present
        location_parts = [part for part in [city, state, country] if part]
        validated["office_locations"] = [", ".join(location_parts)]
    else:
        validated["office_locations"] = []
        validation_warnings.append("Apollo data missing complete location information")

    validated["validation_warnings"] = validation_warnings
    return validated

def diagnose_json_issues(json_str: str) -> Dict:
    diagnosis = {"unclosed_constructs": [], "string_issues": []}
    stack = []
    for i, c in enumerate(json_str):
        if c in '{[':
            stack.append((c, i))
        elif c in '}]':
            if not stack:
                diagnosis["unclosed_constructs"].append(f"Extra {c} at {i}")
                continue
            last = stack.pop()
            if (c == '}' and last[0] != '{') or (c == ']' and last[0] != '['):
                diagnosis["unclosed_constructs"].append(
                    f"Mismatch: {last[0]} at {last[1]} vs {c} at {i}"
                )
    
    in_string = False
    escape = False
    start_pos = 0
    for i, c in enumerate(json_str):
        if c == '"' and not escape:
            if not in_string:
                start_pos = i
                in_string = True
            else:
                in_string = False
        escape = (c == '\\' and not escape)
    
    if in_string:
        diagnosis["string_issues"].append(
            f"Unterminated string starting at {start_pos}: {json_str[start_pos:start_pos+50]}"
        )
    
    return diagnosis

# API Endpoints
@app.get("/prompts", summary="List Available Prompts", description="Returns the list of available search prompts with their indices, titles, and agent IDs.")
async def list_prompts() -> List[Dict]:
    return [{"index": i, "title": p["title"], "agent-ID": p["agent-ID"]} for i, p in enumerate(search_prompts)]

@app.post("/run_prompt", summary="Run a Specific Prompt", description="Runs a specific prompt by index and returns processed results.")
async def run_prompt(request: PromptRequest) -> Dict:
    try:
        prompt_data = search_prompts[request.prompt_index]
    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid prompt index")
    
    if request.custom_message and not request.custom_message.strip():
        raise HTTPException(status_code=400, detail="Custom message cannot be empty if provided")
    
    prompt_content = (
        f"Return at least 15-20 firms + User Request: {request.custom_message} [IMPORTANT]"
        if request.custom_message else
        "Return at least 15-20 firms"
    )
    logger.info(f"Processing prompt with custom message: {request.custom_message}")

    response = await query_perplexity(prompt_content, prompt_data['agent-ID'])
    if not response:
        raise HTTPException(status_code=500, detail="API request failed")

    processed = await process_api_response(response, prompt_data)

    # Validate number of companies
    companies = processed.get("companies", [])
    if len(companies) < 15:
        processed["validation_warnings"].append(
            f"Response returned {len(companies)} companies, expected at least 15"
        )

    # Save results to `prompt_results_collection`
    document = {
        "prompt_index": request.prompt_index,
        "title": prompt_data['title'],
        "custom_message": request.custom_message,
        "prompt_content": prompt_content,
        "raw_response": processed.get("raw_response", []),
        "extracted_companies": companies,
        "validation_warnings": processed.get("validation_warnings", []),
        "timestamp": datetime.utcnow()
    }
    result = await prompt_results_collection.insert_one(document)
    processed["document_id"] = str(result.inserted_id)

    # Upsert companies into `companies_collection`
    for company in processed.get("response", []):
        if "name" in company:
            await companies_collection.update_one(
                {"name": company["name"]},
                {"$set": company},
                upsert=True
            )

    return processed

@app.get("/prompt_history", summary="Get Prompt History", description="Retrieves all saved prompt results")
async def get_prompt_history() -> List[Dict]:
    cursor = prompt_results_collection.find().sort("timestamp", -1)
    results = []
    async for document in cursor:
        document.pop("_id", None)
        results.append(document)
    return results


@app.get("/companies", summary="Get All Companies", description="Retrieves all unique company records from the companies collection")
async def get_all_companies() -> List[Dict]:
    cursor = companies_collection.find({})
    companies = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        companies.append(doc)
    return jsonable_encoder(companies)


@app.post("/extract_companies", summary="Extract Companies from Response", description="Extracts company names from a given JSON response content.")
async def extract_companies_endpoint(content: str) -> CompaniesResponse:
    try:
        json_content = json.loads(sanitize_json_string(content))
        companies = extract_companies(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input content as JSON: {e}")
        companies = []
    return CompaniesResponse(companies=companies)

@app.post("/analyze", summary="Analyze Companies with Claude", description="Analyzes a list of companies using Lyzr agent API for Claude, expecting JSON output with rankings and rationales for merger candidacy based on provided company data.")
async def analyze_companies(request: AnalysisRequest) -> Dict:
    if not request.companies:
        raise HTTPException(status_code=400, detail="No companies provided for analysis")
    companies_dict = [company.dict() for company in request.companies]
    analysis = await analyze_companies_with_claude(companies_dict)
    if not analysis:
        raise HTTPException(status_code=500, detail="Lyzr agent API request failed or returned empty analysis")
    return {"analysis": analysis}

@app.post("/run_merger_search", summary="Run Full Merger Search", description="Executes all prompts, extracts companies, analyzes with Claude, and stores results in MongoDB.")
async def run_merger_search_endpoint() -> MergerSearchResponse:
    batch_tasks = [process_prompt_batch(p) for p in search_prompts]
    batch_results = await asyncio.gather(*batch_tasks)

    results = {}
    all_company_docs = []
    raw_responses = {}

    for result in batch_results:
        if "error" in result:
            logger.error(f"Failed batch item: {result.get('title', 'Unknown')}")
            continue

        key = result["title"]
        results[key] = {
            "raw_response": result["response"],
            "extracted_companies": result["companies"],
            "validation_warnings": result["validation_warnings"]
        }

        raw_responses[key] = json.dumps(result["response"], indent=2)

        for company in result["response"]:
            if "name" in company and company["name"]:
                # Add timestamp to each company entry
                company_doc = company.copy()
                company_doc["timestamp"] = datetime.utcnow()
                all_company_docs.append(company_doc)

    # Save results to merger_search_collection
    merger_document = {
        "results": results,
        "all_companies": list({c['name'] for c in all_company_docs}),
        "claude_analysis": {},
        "timestamp": datetime.utcnow()
    }
    await merger_search_collection.insert_one(merger_document)

    # Upsert each company into companies_collection (replace if exists by name)
    if all_company_docs:
        bulk_ops = []
        for company in all_company_docs:
            bulk_ops.append(
                ReplaceOne(
                    {"name": company["name"]},  # Filter
                    company,  # New doc
                    upsert=True  # Replace or insert
                )
            )
        if bulk_ops:
            await companies_collection.bulk_write(bulk_ops)

    # Analyze companies using Claude
    analysis_task = analyze_with_claude(list({c['name'] for c in all_company_docs}), raw_responses)
    claude_analysis = await analysis_task

    # Update merger document with analysis result
    await merger_search_collection.update_one(
        {"_id": merger_document["_id"]},
        {"$set": {"claude_analysis": claude_analysis}}
    )

    return MergerSearchResponse(
        results={**results, "claude_analysis": claude_analysis},
        message="Merger search completed and saved to MongoDB"
    )

@app.post("/redo_search", summary="Redo Merger Search", description="Re-runs the full merger search process, overwriting existing results.")
async def redo_search_endpoint() -> MergerSearchResponse:
    # Clear existing files if they exist
    csv_file = Path("identified_companies.csv")
    json_file = Path("merger_search_results.json")
    if csv_file.exists():
        csv_file.unlink()
    if json_file.exists():
        json_file.unlink()

    # Run the full merger search
    return await run_merger_search_endpoint()

@app.get("/fetch_existing_items")
async def fetch_existing_items() -> Dict:
    # Get latest merger search result
    latest_search = await merger_search_collection.find_one(
        sort=[("timestamp", -1)]
    )
    
    if not latest_search:
        return {"results": {}, "message": "No existing results found"}
    
    # Remove MongoDB ID before returning
    latest_search.pop("_id", None)
    return {"results": latest_search, "message": "Successfully retrieved existing results"}

async def save_to_csv(companies: List[str]):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: pd.DataFrame({"Company": companies}).to_csv("identified_companies.csv", index=False)
    )

async def save_to_json(data: Dict):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: json.dump(data, open('merger_search_results.json', 'w'), indent=2)
    )

@app.get("/results", summary="Get Saved Results", description="Retrieves the latest merger search results from MongoDB.")
async def get_results() -> Dict:
    latest_result = await merger_search_collection.find_one(
        {}, 
        sort=[("timestamp", -1)]
    )

    if not latest_result:
        raise HTTPException(status_code=404, detail="No merger search results found")

    # Optionally remove the MongoDB ObjectId if not needed in response
    latest_result.pop("_id", None)

    return latest_result

@app.get("/download_csv", summary="Download Companies CSV", description="Downloads the CSV file containing the list of identified companies.")
async def download_csv() -> FileResponse:
    csv_file = Path("identified_companies.csv")
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    return FileResponse(csv_file, filename="identified_companies.csv")

@app.get("/download_json", summary="Download Results JSON", description="Downloads the JSON file containing the full merger search results.")
async def download_json() -> FileResponse:
    json_file = Path("merger_search_results.json")
    if not json_file.exists():
        raise HTTPException(status_code=404, detail="JSON file not found")
    return FileResponse(json_file, filename="merger_search_results.json")

@app.post("/enrich_company", summary="Enrich Company Details", description="Enriches company details using the Apollo API, given a company domain.")
async def enrich_company_endpoint(request: EnrichCompanyRequest) -> Dict:
    if not request.company_domain.strip():
        raise HTTPException(status_code=400, detail="Company domain cannot be empty")
    result = await enrich_company(request.company_domain)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# Fetch Company Details from Apollo
# Fetch Company Details from Apollo
@app.post("/fetch_company_apollo", summary="Fetch Company Details from Apollo", description="Fetches detailed company information from Apollo API for merger and acquisition analysis, given a company domain.")
async def fetch_company_apollo(request: EnrichCompanyRequest) -> Dict:
    if not request.company_domain.strip():
        raise HTTPException(status_code=400, detail="Company domain cannot be empty")
    
    result = await enrich_company(request.company_domain)
    if result is None:
        logger.error(f"Apollo API returned None for domain {request.company_domain}")
        raise HTTPException(status_code=500, detail="Apollo API returned invalid response")
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=f"Apollo API error: {result['error']}")
    
    Path("api_logs").mkdir(exist_ok=True)
    with open("api_logs/apollo_enrich.log", "a") as f:
        f.write(f"{time.time()}: {json.dumps(result, indent=2)}\n")
    
    return {
        "company": result,
        "message": f"Successfully retrieved Apollo data for {request.company_domain}"
    }

@app.post("/fetch_company_perplexity", summary="Fetch Company Details from Perplexity via Lyzr", description="Fetches detailed company information from Perplexity via Lyzr agent for merger and acquisition analysis, given a company name and domain.")
async def fetch_company_perplexity(request: FetchCompanyPerplexityRequest) -> Dict:
    if not request.company_name.strip() or not request.company_domain.strip():
        raise HTTPException(status_code=400, detail="Company name and domain cannot be empty")
    
    prompt = f"""
Provide detailed information about the retail consulting firm {request.company_name} ({request.company_domain}) for merger and acquisition analysis:

Target Firm:
- Must be headquartered in North America
- Annual revenue: Strictly $10-500M (exclude if outside this range)
- Specializations: Must include Product Development, Sourcing Strategy, or Supply Chain Optimization
- Primary client base: Enterprise retailers with revenue >$500M (e.g., Walmart, Target)

Research Methodology:
- Source data from the company website ({request.company_domain}), industry reports (e.g., IBISWorld, Retail Dive), and LinkedIn profiles
- Validate all data with at least two credible sources (e.g., company website, SEC filings, LinkedIn, industry reports)
- Exclude if data is incomplete, unvalidated, or if the firm does not meet all criteria (revenue, headquarters, specializations, client base)

Evaluation Criteria:
- Financial Performance: Provide estimated revenue and growth rate (e.g., CAGR over 3 years)
- Operational Scale: Provide employee count as a range (e.g., 50-75 employees)
- Client Portfolio: List key retail clients (must be enterprise retailers with revenue >$500M, validated by testimonials or public records)
- Leadership: Identify key executives with names and titles
- Merger Fit: Describe synergies (strategic, operational, cultural) with a hypothetical retail consulting acquirer

Output Requirements:
- Return a JSON object with the following structure:
  ```json
  {{
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "key_clients": ["<string>"],
    "leadership": [
      {{
        "name": "<string>",
        "title": "<string>"
      }}
    ],
    "merger_synergies": "<string>",
    "Industries": "<string>",
    "Services": "<string>",
    "Broad Category": "<string>",
    "Ownership": "<string> e.g., Private, JV, Partnership, Public",
    "sources": ["<string>"]
  }}
  ```
- Include at least 2 credible sources (e.g., company website, LinkedIn, industry reports)
- Return JSON only, with no narrative text or comments
- Return an empty object {{}} if data is unavailable, cannot be validated by at least two sources, or if the firm does not meet all criteria
"""
    response = await query_perplexity(prompt, "6817b49c2f31429a00a39d39")  # Using first agent-ID
    if not response:
        raise HTTPException(status_code=500, detail="Perplexity API request failed")
    
    raw_content = response.get('response', '')
    logger.debug(f"Raw Perplexity API response for {request.company_name}:\n{raw_content[:2000]}")
    Path("api_logs").mkdir(exist_ok=True)
    log_file = f"api_logs/perplexity_{time.time()}.json"
    with open(log_file, "w") as f:
        f.write(raw_content)
    
    sanitized = sanitize_json_string(raw_content)
    try:
        json_content = json.loads(sanitized)
    except json.JSONDecodeError as e:
        logger.error(f"Critical JSON failure for {request.company_name}: {e}")
        return {
            "company": {},
            "message": "Failed to parse Perplexity response",
            "validation_warnings": [f"Invalid JSON response: {str(e)}"]
        }
    
    company_data = json_content if isinstance(json_content, dict) else {}
    validation_warnings = []
    
    if not company_data:
        validation_warnings.append(f"No valid data returned for {request.company_name}")
    else:
        required_fields = ["name", "domain_name", "estimated_revenue", "employee_count", "sources"]
        missing_fields = [field for field in required_fields if not company_data.get(field)]
        if missing_fields:
            validation_warnings.append(f"Company {request.company_name} missing fields: {missing_fields}")
            company_data = {}
        else:
            revenue_str = company_data.get("estimated_revenue", "")
            try:
                revenue = float(revenue_str.replace("$", "").replace("M", ""))
                if not (10 <= revenue <= 20):
                    validation_warnings.append(f"Company {request.company_name} revenue {revenue}M outside $10-500M range")
                    company_data = {}
            except (ValueError, TypeError):
                validation_warnings.append(f"Invalid revenue format for {request.company_name}: {revenue_str}")
                company_data = {}
            if len(company_data.get("sources", [])) < 2:
                validation_warnings.append(f"Company {request.company_name} has fewer than 2 credible sources")
                company_data = {}
    
    if company_data:
        logger.info(f"Fetching Apollo data for {request.company_name} with domain {request.company_domain}")
        await asyncio.sleep(0.5)
        apollo_data = await enrich_company(request.company_domain)
        if "error" not in apollo_data:
            company_data = validate_company_data(company_data, apollo_data)
            validation_warnings.extend(company_data.get("validation_warnings", []))
        else:
            company_data["validation_warnings"] = [f"Apollo API failed: {apollo_data['error']}"]
            validation_warnings.append(f"Apollo API failed for {request.company_name}: {apollo_data['error']}")
    
    return {
        "company": company_data,
        "message": f"Successfully retrieved Perplexity data for {request.company_name}" if company_data else f"No valid data for {request.company_name}",
        "validation_warnings": validation_warnings
    }

@app.get('/health', summary="Health Check", description="Checks the health status of the API.")
async def health_check():
    return {"status": "ok"}