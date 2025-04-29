from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
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
LYZR_PERPLEXITY_AGENT_ID = os.getenv("LYZR_PERPLEXITY_AGENT_ID")
LYZR_PERPLEXITY_SESSION_ID = os.getenv("LYZR_PERPLEXITY_SESSION_ID")
LYZR_CLAUDE_AGENT_ID = os.getenv("LYZR_CLAUDE_AGENT_ID")
LYZR_CLAUDE_SESSION_ID = os.getenv("LYZR_CLAUDE_SESSION_ID")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")

# Validate required environment variables
required_env_vars = {
    "LYZR_AGENT_URL": LYZR_AGENT_URL,
    "LYZR_API_KEY": LYZR_API_KEY,
    "LYZR_USER_ID": LYZR_USER_ID,
    "LYZR_PERPLEXITY_AGENT_ID": LYZR_PERPLEXITY_AGENT_ID,
    "LYZR_PERPLEXITY_SESSION_ID": LYZR_PERPLEXITY_SESSION_ID,
    "LYZR_CLAUDE_AGENT_ID": LYZR_CLAUDE_AGENT_ID,
    "LYZR_CLAUDE_SESSION_ID": LYZR_CLAUDE_SESSION_ID,
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

import ast

def sanitize_json_string(json_str: any) -> str:
    """
    Comprehensive JSON sanitization with multi-stage repair capabilities
    Handles smart quotes, unescaped characters, and complex code blocks
    """
    # Initial cleaning and conversion
    json_str = str(json_str).strip()
    if not json_str:
        return "{}"

    # Remove all ASCII control characters except newline/tab
    json_str = re.sub(r'[\x00-\x09\x0b-\x1f\x7f]', '', json_str)
    
    # Handle mixed code block formats
    json_str = re.sub(
        r'(?:```(?:json)?|~~~)(.*?)(?:```|~~~)', 
        lambda m: m.group(1).strip(), 
        json_str, 
        flags=re.DOTALL
    )

    # Convert smart quotes and other Unicode characters
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '\u2028': ' ', '\u2029': ' ', '\ufeff': ''
    }
    for bad, good in replacements.items():
        json_str = json_str.replace(bad, good)

    # Fix multiline strings and unescaped quotes
    json_str = re.sub(
        r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'',  # Fixed single quote handling
        lambda m: m.group(0).replace('\n', '\\n'),
        json_str,
        flags=re.DOTALL
    )

    # Stage 1: Attempt direct parse
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        error_pos = e.pos
        logger.warning(f"Initial parse failed at {error_pos}: {e}")

    # Stage 2: Context-aware repair
    try:
        # Handle common unterminated strings
        if "Unterminated string" in str(e):
            quote_char = json_str[error_pos-1]
            end_quote = json_str.find(quote_char, error_pos)
            if end_quote == -1:
                json_str += quote_char
                
        # Balance braces/brackets using stack logic
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

        # Final validation
        return json.dumps(json.loads(json_str), indent=None)
    except Exception:
        pass

    # Stage 3: Safe fallback using literal_eval
    try:
        parsed = ast.literal_eval(json_str)
        return json.dumps(parsed)
    except Exception as e:
        logger.error(f"Literal eval fallback failed: {e}")

    # Final fallback: Wrap in array if root object
    if json_str.startswith('{'):
        return f'[{json_str}]'
    return json_str

# Retry decorator for API calls
from functools import wraps
import time

def retry_api_call(max_attempts: int = 3, delay: float = 2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if result and isinstance(result.get("response"), str):
                        try:
                            json.loads(sanitize_json_string(result["response"]))
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON on attempt {attempt + 1}")
                    logger.warning(f"API call returned None or invalid response on attempt {attempt + 1}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
            logger.error(f"API call failed after {max_attempts} attempts")
            return None
        return wrapper
    return decorator


def async_retry(max_attempts: int = 3, delay: float = 2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if result and isinstance(result.get("response"), str):
                        try:
                            json.loads(sanitize_json_string(result["response"]))
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON on attempt {attempt + 1}")
                    logger.warning(f"API call returned None or invalid response on attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)  # Use async sleep
            logger.error(f"API call failed after {max_attempts} attempts")
            return None
        return wrapper
    return decorator


# Search prompts with stricter validation
search_prompts = [
    {
        "title": "Initial Target Identification",
        "content": """Identify boutique retail consulting firms for merger potential with the following criteria:

Target Firms:
- Headquartered in North America
- Annual revenue: Strictly $10-20M (exclude firms with revenue outside this range)
- Specializations: Product Development, Sourcing Strategy, Supply Chain Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 6 firms meeting all criteria
- Cross-reference data from company websites, industry reports (e.g., Retail Dive, Consulting.us), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, client testimonials, and case studies; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-20M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations
- Client Portfolio: Key retail clients, average contract value
- Leadership: Key executives, their expertise
- Technology: Tools used (e.g., SAP, Tableau)
- Merger Fit: Synergies, cultural alignment, integration challenges

Output Requirements:
- Return a JSON array of 6 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "profitability": "<string, e.g., 20% EBITDA margin>",
      "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "office_locations": ["<string>", ],
      "key_clients": ["<string>", ],
      "average_contract_value": "<string, e.g., $500K>",
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>",
          "experience": "<string, e.g., 15 years in retail consulting>"
        },
        
      ],
      "primary_domains": ["<string>", ],
      "proprietary_methodologies": "<string>",
      "technology_tools": ["<string>", ],
      "competitive_advantage": "<string>",
      "merger_synergies": "<string>",
      "cultural_alignment": "<string>",
      "integration_challenges": "<string>",
      "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
      "sources": ["<string>", ]
    },
  ]
  ```
- Include 2-3 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, mark as 'Data not available' and exclude from results
- Return JSON only, with no narrative text, explanations, or inline comments (e.g., // )
- Ensure all firms meet the $10-20M revenue criterion""",
        "sources": [
            "https://www.consulting.us/rankings/top-consulting-firms-in-the-us-by-industry-expertise/retail",
            "https://managementconsulted.com/boutique-consulting-firms-in-us/",
            "https://www.g-co.agency/insights/best-retail-consulting-firms-to-work-with",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Market Analysis and Competitive Landscape",
        "content": """Analyze the competitive landscape of boutique retail consulting firms for merger potential:

Target Firms:
- Headquartered in North America
- Annual revenue: Strictly $10-20M (exclude firms with revenue outside this range)
- Specializations: Product Development, Sourcing Strategy, Procurement Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 5 firms meeting all criteria
- Cross-reference data from industry reports (e.g., IBISWorld, Gartner), financial databases (e.g., PitchBook), company websites, and LinkedIn
- Validate through client testimonials, press releases, and public records; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-20M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations
- Client Portfolio: Key retail clients, average contract value
- Leadership: Key executives, their expertise
- Technology: Tools used (e.g., AI analytics, supply chain software)
- Competitive Positioning: Market share, competitive advantages
- Merger Fit: Synergies, cultural alignment, integration challenges

Output Requirements:
- Return a JSON object with 5 firms in the following structure:
  ```json
  {
    [
      {
        "rank": <integer>,
        "name": "<string>",
        "domain_name": "<string>",
        "estimated_revenue": "<string, e.g., $15M>",
        "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
        "profitability": "<string, e.g., 20% EBITDA margin>",
        "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
        "employee_count": "<string, e.g., 50-75 employees>",
        "office_locations": ["<string>", ],
        "key_clients": ["<string>", ],
        "average_contract_value": "<string, e.g., $500K>",
        "leadership": [
          {
            "name": "<string>",
            "title": "<string>",
            "experience": "<string, e.g., 15 years in retail consulting>"
          },
        ],
        "primary_domains": ["<string>", ],
        "proprietary_methodologies": "<string>",
        "technology_tools": ["<string>", ],
        "competitive_advantage": "<string>",
        "merger_synergies": "<string>",
        "cultural_alignment": "<string>",
        "integration_challenges": "<string>",
        "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
        "sources": ["<string>", ]
      },
    ]
  }
  ```
- Include 2-3 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, mark as 'Data not available' and exclude from results
- Return JSON only, with no narrative text, explanations, or inline comments (e.g., // )
- Ensure all firms meet the $10-20M revenue criterion""",
        "sources": [
            "https://www.ibisworld.com/",
            "https://www.gartner.com/",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/",
            "https://www.retaildive.com/"
        ]
    },
    {
        "title": "Sourcing and Procurement Specialists",
        "content": """Conduct a comprehensive survey of boutique procurement strategy consulting firms for merger potential:

Target Firms:
- Headquartered in North America
- Annual revenue: Strictly $10-20M (exclude firms with revenue outside this range, e.g., Accenture, Deloitte)
- Specializations: Strategic Sourcing, Global Procurement, Retail Supply Chain Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify exactly 7 firms meeting all criteria
- Cross-reference data from company websites, industry reports (e.g., Procurement Leaders, CIPS), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, client testimonials, and case studies; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-20M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations, global sourcing reach
- Client Portfolio: Key retail clients, average contract value
- Leadership: Key executives, their procurement expertise
- Technology: Tools for procurement optimization (e.g., SAP Ariba, Coupa)
- Merger Fit: Synergies, cultural alignment, integration challenges

Output Requirements:
- Return a JSON object with exactly 7 firms in the following structure:
  ```json
  {
    [
      {
        "name": "<string>",
        "domain_name": "<string>",
        "estimated_revenue": "<string, e.g., $15M>",
        "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
        "profitability": "<string, e.g., 20% EBITDA margin>",
        "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
        "employee_count": "<string, e.g., 50-75 employees>",
        "office_locations": ["<string>", ],
        "key_clients": ["<string>", ],
        "average_contract_value": "<string, e.g., $500K>",
        "leadership": [
          {
            "name": "<string>",
            "title": "<string>",
            "experience": "<string, e.g., 15 years in procurement>"
          },
        ],
        "primary_domains": ["<string>", ],
        "proprietary_methodologies": "<string>",
        "technology_tools": ["<string>", ],
        "competitive_advantage": "<string>",
        "merger_synergies": "<string>",
        "cultural_alignment": "<string>",
        "integration_challenges": "<string>",
        "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
        "technological_enablement_score": "<string, e.g., High, based on AI adoption>",
        "global_sourcing_reach": "<string, e.g., Strong in Asia>",
        "sources": ["<string>", ]
      },
    ]
  }
  ```
- Include 2-3 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, mark as 'Data not available' and exclude from results
- Return JSON only, with no narrative text, explanations, or inline comments (e.g., // )
- Ensure all firms meet the $10-20M revenue criterion""",
        "sources": [
            "https://www.supplychaindigital.com/technology/top-10-procurement-consulting-firms",
            "https://www.consultancy.uk/sectors/procurement-consulting",
            "https://www.procurementleaders.com/industry-insight/consulting-rankings",
            "https://www.cips.org/knowledge/procurement-strategy/",
            "https://www.supplychaindive.com/news/top-procurement-consulting-firms/",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Product Development Expertise",
        "content": """Identify boutique retail consulting firms with expertise in product development for merger potential:

Target Firms:
- Headquartered in North America
- Annual revenue: Strictly $10-20M (exclude firms with revenue outside this range)
- Specializations: Concept Development, Product Optimization, Launch Strategy, Post-Launch Performance Analysis
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 4 firms meeting all criteria
- Cross-reference data from company websites, industry reports (e.g., Retail Dive, PDMA), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, client testimonials, and case studies; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-20M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations
- Client Portfolio: Key retail clients, average contract value
- Leadership: Key executives, their product development expertise
- Technology: Tools used (e.g., PLM software, Tableau)
- Merger Fit: Synergies, cultural alignment, integration challenges

Output Requirements:
- Return a JSON array of 4 firms in the following structure:
  ```json
  [
    {
      "name": "<string>",
      "domain_name": "<string>",
      "estimated_revenue": "<string, e.g., $15M>",
      "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
      "profitability": "<string, e.g., 20% EBITDA margin>",
      "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
      "employee_count": "<string, e.g., 50-75 employees>",
      "office_locations": ["<string>", ],
      "key_clients": ["<string>", ],
      "average_contract_value": "<string, e.g., $500K>",
      "leadership": [
        {
          "name": "<string>",
          "title": "<string>",
          "experience": "<string, e.g., 15 years in product development>"
        },
      ],
      "primary_domains": ["<string>", ],
      "proprietary_methodologies": "<string>",
      "technology_tools": ["<string>", ],
      "competitive_advantage": "<string>",
      "merger_synergies": "<string>",
      "cultural_alignment": "<string>",
      "integration_challenges": "<string>",
      "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
      "innovation_metrics": "<string, e.g., 80% launch success rate>",
      "sources": ["<string>", ]
    },
    
  ]
  ```
- Include 2-3 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, mark as 'Data not available' and exclude from results
- Return JSON only, with no narrative text, explanations, or inline comments (e.g., // )
- Ensure all firms meet the $10-20M revenue criterion""",
        "sources": [
            "https://www.retaildive.com/",
            "https://www.pdma.org/",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/",
            "https://www.g-co.agency/insights/best-retail-consulting-firms-to-work-with"
        ]
    },
    {
        "title": "Supply Chain Specialists",
        "content": """Identify boutique retail consulting firms with expertise in supply chain optimization for merger potential:

Target Firms:
- Headquartered in North America
- Annual revenue: Strictly $10-20M (exclude firms with revenue outside this range)
- Specializations: Supply Chain Optimization, Vendor Assessment/Evaluation, Digital Supply Chain Transformation
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Identify 4 firms meeting all criteria
- Cross-reference data from company websites, industry reports (e.g., Gartner, Supply Chain Dive), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, client testimonials, and case studies; data must be confirmed by at least two credible sources listed below
- Exclude firms with incomplete data or revenue outside $10-20M

Evaluation Criteria:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations
- Client Portfolio: Key retail clients, average contract value
- Leadership: Key executives, their supply chain expertise
- Technology: Tools used (e.g., Blue Yonder, Kinaxis)
- Merger Fit: Synergies, cultural alignment, integration challenges

Output Requirements:
- Return a JSON object with 4 firms in the following structure:
  ```json
  {
    [
      {
        "name": "<string>",
        "domain_name": "<string>",
        "estimated_revenue": "<string, e.g., $15M>",
        "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
        "profitability": "<string, e.g., 20% EBITDA margin>",
        "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
        "employee_count": "<string, e.g., 50-75 employees>",
        "office_locations": ["<string>", ],
        "key_clients": ["<string>", ],
        "average_contract_value": "<string, e.g., $500K>",
        "leadership": [
          {
            "name": "<string>",
            "title": "<string>",
            "experience": "<string, e.g., 15 years in supply chain>"
          },
        ],
        "primary_domains": ["<string>", ],
        "proprietary_methodologies": "<string>",
        "technology_tools": ["<string>", ],
        "competitive_advantage": "<string>",
        "merger_synergies": "<string>",
        "cultural_alignment": "<string>",
        "integration_challenges": "<string>",
        "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
        "digital_transformation_score": "<string, e.g., High, based on AI adoption>",
        "sustainability_performance": "<string, e.g., 30% reduction in carbon footprint>",
        "sources": ["<string>", ]
      },
    ]
  }
  ```
- Include 2-3 credible sources per firm
- If data is unavailable or cannot be validated by at least two sources, mark as 'Data not available' and exclude from results
- Return JSON only, with no narrative text, explanations, or inline comments (e.g., // )
- Ensure all firms meet the $10-20M revenue criterion""",
        "sources": [
            "https://www.gartner.com/",
            "https://www.supplychaindive.com/",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/",
            "https://www.walmartlogisticsreport.com/"
        ]
    }
]

# Pydantic models for request/response validation
class PromptRequest(BaseModel):
    prompt_index: int

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

# Apollo API enrichment function
async def enrich_company(company_domain: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.apollo.io/api/v1/organizations/enrich",
                params={"domain": company_domain},
                headers={"x-api-key": APOLLO_API_KEY}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Apollo API error: {str(e)}")
            return {"error": str(e)}

# Validate and update company data with Apollo API
def validate_company_data(perplexity_data: Dict, apollo_data: Dict) -> Dict:
    validated = perplexity_data.copy()
    validation_warnings = []

    # Update revenue
    if apollo_data.get("annual_revenue_printed"):
        apollo_revenue = apollo_data["annual_revenue_printed"]
        perplexity_revenue = perplexity_data.get("estimated_revenue", "")
        try:
            apollo_revenue_val = float(apollo_revenue.replace("$", "").replace("M", ""))
            perplexity_revenue_val = float(perplexity_revenue.replace("$", "").replace("M", ""))
            if abs(apollo_revenue_val - perplexity_revenue_val) > 5:
                validation_warnings.append(
                    f"Revenue discrepancy: Perplexity={perplexity_revenue}, Apollo={apollo_revenue}"
                )
            validated["estimated_revenue"] = f"${apollo_revenue_val}M"
        except (ValueError, TypeError):
            validation_warnings.append(
                f"Invalid revenue format: Perplexity={perplexity_revenue}, Apollo={apollo_revenue}"
            )

    # Update employee count
    if apollo_data.get("estimated_num_employees"):
        apollo_employees = apollo_data["estimated_num_employees"]
        perplexity_employees = perplexity_data.get("employee_count", "")
        try:
            apollo_count = int(apollo_employees)
            perplexity_count = int(perplexity_employees.split("-")[0].replace("employees", "").strip()) if "-" in perplexity_employees else int(perplexity_employees.replace("employees", "").strip())
            if abs(apollo_count - perplexity_count) > 20:
                validation_warnings.append(
                    f"Employee count discrepancy: Perplexity={perplexity_employees}, Apollo={apollo_employees}"
                )
            validated["employee_count"] = f"{apollo_employees} employees"
        except (ValueError, TypeError):
            validation_warnings.append(
                f"Invalid employee count format: Perplexity={perplexity_employees}, Apollo={apollo_employees}"
            )

    # Update office locations
    if apollo_data.get("city") and apollo_data.get("state"):
        apollo_location = f"{apollo_data['city']}, {apollo_data['state']}"
        perplexity_locations = perplexity_data.get("office_locations", [])
        if apollo_location not in perplexity_locations:
            validation_warnings.append(
                f"Location discrepancy: Perplexity={perplexity_locations}, Apollo={apollo_location}"
            )
            validated["office_locations"] = [apollo_location] + perplexity_locations

    # Update technology tools
    if apollo_data.get("technology_names"):
        apollo_tech = apollo_data["technology_names"]
        perplexity_tech = perplexity_data.get("technology_tools", [])
        validated["technology_tools"] = list(set(apollo_tech + perplexity_tech))

    # Update primary domains
    if apollo_data.get("keywords"):
        apollo_domains = [kw for kw in apollo_data["keywords"] if kw in perplexity_data.get("primary_domains", [])]
        validated["primary_domains"] = list(set(apollo_domains + perplexity_data.get("primary_domains", [])))

    validated["validation_warnings"] = validation_warnings
    return validated

# API Functions
@async_retry(max_attempts=3, delay=2.0)
async def query_lyzr_agent(agent_id: str, session_id: str, prompt: str) -> Optional[Dict]:
    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": agent_id,
        "session_id": session_id,
        "message": prompt
    }
    try:
        response = await async_client.post(
            LYZR_AGENT_URL,
            json=payload,
            headers=LYZR_AGENT_HEADERS
        )
        response.raise_for_status()
        json_response = response.json()
        raw_response = json_response.get("response", "{}")
        print("response",response.json())
        logger.info(f"API response size: {len(str(json_response))} bytes")
        json_response["response"] = sanitize_json_string(raw_response)
        return json_response
    except Exception as e:
        logger.error(f"Lyzr agent API error: {e}")
        raise


async def process_prompt_batch(prompt_data: Dict) -> Dict:
    try:
        response = await query_perplexity(prompt_data['content'])
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


async def query_perplexity(prompt: str) -> Optional[Dict]:
    return await query_lyzr_agent(LYZR_PERPLEXITY_AGENT_ID, LYZR_PERPLEXITY_SESSION_ID, prompt)

async def query_claude(prompt: str) -> Optional[Dict]:
    return await query_lyzr_agent(LYZR_CLAUDE_AGENT_ID, LYZR_CLAUDE_SESSION_ID, prompt)

def extract_companies(json_content: any) -> List[str]:
    """Deep search for company data in complex JSON structures"""
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

async def analyze_with_claude(companies: List[str], raw_responses: Dict[str, str]) -> Optional[Dict]:
    full_context = "\n\n".join([f"### {title}\n{content}" for title, content in raw_responses.items()])
    company_list = "\n".join([f"- {company}" for company in companies])
    print("Full context", full_context)
    print("company list", company_list)
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

Analysis Methodology:
1. Financial Health: Calculate weighted scores based on revenue CAGR, EBITDA margins, and valuation multiples
2. Strategic Fit: Evaluate service complementarity, client overlap, and market penetration
3. Operational Compatibility: Assess office locations, employee scale, and technology alignment
4. Leadership and Innovation: Review leadership experience, proprietary methodologies, and innovation metrics
5. Cultural and Integration Risks: Analyze cultural alignment and potential integration challenges

Constraints:
- Only analyze the companies explicitly listed in the provided company_list
- Do not include or evaluate any companies not present in the company_list
- If a company in the company_list lacks sufficient data, include it in the output with scores based on available data or reasonable estimates, noting assumptions in the rationale

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
          "rationale": "<string, brief explanation of ranking, including any assumptions for missing data>"
        }
      ],
      "recommendations": [
        {
          "name": "<string>",
          "merger_potential": "<string, e.g., High/Medium/Low>",
          "key_synergies": ["<string>"],
          "potential_risks": ["<string>"]
        }
      ],
      "summary": "<string, 2-3 sentences summarizing the analysis of the provided companies>"
    }
  }
  ```
- Include only the companies from the provided company_list in the rankings and recommendations
- Rank all provided firms based on weighted scores
- Provide actionable recommendations for each listed firm
- Include a concise summary of findings focused on the listed companies
- Return JSON only, with no narrative text, explanations, or inline comments
- If data is missing for a listed company, use reasonable estimates and note assumptions in the rationale
- Ensure the company_list is populated with valid company names before analysis
- Validate that full_context contains sufficient data for analysis, and handle empty or malformed inputs gracefully
"""

    print("Prompt", analysis_prompt)
    response = await query_claude(analysis_prompt+""+company_list+""+full_context)
    if response:
        try:
            json_content = json.loads(sanitize_json_string(response.get("response", "{}")))
            return json_content.get("analysis", {})
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            return {}
    return {}


# API Endpoints
@app.get("/prompts", summary="List Available Prompts", description="Returns the list of available search prompts.")
async def list_prompts() -> List[Dict]:
    return [{"index": i, "title": p["title"]} for i, p in enumerate(search_prompts)]

@app.post("/run_prompt")
async def run_prompt(request: PromptRequest) -> Dict:
    try:
        prompt_data = search_prompts[request.prompt_index]
    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid prompt index")
    
    response = await query_perplexity(prompt_data['content'])
    if not response:
        raise HTTPException(status_code=500, detail="API request failed")

    processed = await process_api_response(response, prompt_data)
    return processed

async def process_api_response(response: Dict, prompt_data: Dict) -> Dict:
    """Process Perplexity response and validate with Apollo API"""
    raw_content = response.get('response', '')
    
    # Forensic logging for Perplexity response
    logger.debug(f"Raw Perplexity API response:\n{raw_content[:2000]}")
    Path("api_logs").mkdir(exist_ok=True)
    log_file = f"api_logs/{time.time()}.json"
    with open(log_file, "w") as f:
        f.write(raw_content)
    
    # Sanitize and parse response
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
    
    # Validate company data
    validated_companies = []
    validation_warnings = []
    for company in json_content if isinstance(json_content, list) else []:
        if not isinstance(company, dict):
            continue
        # Check required fields
        required_fields = ["name", "domain_name", "estimated_revenue", "employee_count", "office_locations"]
        missing_fields = [field for field in required_fields if not company.get(field)]
        if missing_fields:
            validation_warnings.append(f"Company {company.get('name', 'unknown')} missing fields: {missing_fields}")
            continue
        # Validate revenue range ($10-20M)
        revenue_str = company.get("estimated_revenue", "")
        try:
            revenue = float(revenue_str.replace("$", "").replace("M", ""))
            if not (10 <= revenue <= 20):
                validation_warnings.append(f"Company {company['name']} revenue {revenue}M outside $10-20M range")
                continue
        except (ValueError, TypeError):
            validation_warnings.append(f"Invalid revenue format for {company['name']}: {revenue_str}")
            continue

        # Validate with Apollo API
        logger.info(f"Fetching Apollo data for {company['name']} with domain {company['domain_name']}")
        await asyncio.sleep(0.5)  # Non-blocking sleep to avoid Apollo rate limiting
        apollo_data = await enrich_company(company.get("domain_name"))
        if "error" not in apollo_data:
            validated_company = validate_company_data(company, apollo_data)
            validated_companies.append(validated_company)
            validation_warnings.extend(validated_company.get("validation_warnings", []))
        else:
            validation_warnings.append(f"Apollo API failed for {company['name']}: {apollo_data['error']}")

    # Extract company names
    companies = [company["name"] for company in validated_companies]

    return {
        "title": prompt_data["title"],
        "response": validated_companies,
        "companies": companies,
        "sources": prompt_data["sources"],
        "validation_warnings": validation_warnings
    }

def diagnose_json_issues(json_str: str) -> Dict:
    """Detailed JSON diagnostics"""
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

@app.post("/extract_companies", summary="Extract Companies from Response", description="Extracts company names from a given JSON response content.")
async def extract_companies_endpoint(content: str) -> CompaniesResponse:
    try:
        json_content = json.loads(sanitize_json_string(content))
        companies = extract_companies(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input content as JSON: {e}")
        companies = []
    return CompaniesResponse(companies=companies)

@app.post("/analyze", summary="Analyze Companies with Claude", description="Analyzes a list of companies using Lyzr agent API for Claude, expecting JSON output, for merger candidacy based on provided responses.")
async def analyze_companies(request: AnalysisRequest) -> Dict:
    analysis = await analyze_with_claude(request.companies, request.raw_responses)
    if not analysis:
        raise HTTPException(status_code=500, detail="Lyzr agent API request failed or returned empty analysis")
    return {"analysis": analysis}

@app.post("/run_merger_search", summary="Run Full Merger Search", description="Executes the full merger search process, including running all prompts, extracting companies, saving results, and analyzing with Claude via Lyzr agent API, returning all responses as JSON objects.")
async def run_merger_search_endpoint() -> MergerSearchResponse:
    # Create processing tasks for all prompts
    batch_tasks = [process_prompt_batch(p) for p in search_prompts]
    
    # Run all prompts in parallel
    batch_results = await asyncio.gather(*batch_tasks)

    # Process results
    results = {}
    all_companies = []
    raw_responses = {}

    for result in batch_results:
        if "error" in result:
            logger.error(f"Failed batch item: {result['title']}")
            continue
            
        key = result["title"]
        results[key] = {
            "raw_response": result["response"],
            "extracted_companies": result["companies"],
            "validation_warnings": result["validation_warnings"]
        }
        all_companies.extend(result["companies"])
        raw_responses[key] = json.dumps(result["response"], indent=2)

    # Deduplicate companies
    all_companies = sorted(list(set(all_companies)))

    # Save results concurrently
    io_tasks = [
        save_to_csv(all_companies),
        save_to_json(results)
    ]
    await asyncio.gather(*io_tasks)

    # Parallel analysis
    analysis_task = analyze_with_claude(all_companies, raw_responses)
    claude_analysis = await analysis_task

    return MergerSearchResponse(
        results={**results, "claude_analysis": claude_analysis},
        message="Merger search completed with parallel processing"
    )


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

@app.get("/results", summary="Get Saved Results", description="Retrieves the saved merger search results from the JSON file.")
async def get_results() -> Dict:
    result_file = Path("merger_search_results.json")
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    with open(result_file, 'r') as f:
        results = json.load(f)
    return results

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

@app.post("/enrich_company", summary="Enrich Company Details", description="Enriches company details using the Apollo API, given a company domain (e.g., clarkstonconsulting.com). Logs the response to apollo_enrich.log.")
async def enrich_company_endpoint(request: EnrichCompanyRequest) -> Dict:
    if not request.company_domain.strip():
        raise HTTPException(status_code=400, detail="Company domain cannot be empty")
    result = enrich_company(request.company_domain)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get('/health', summary="Health Check", description="Checks the health status of the API.")
async def health_check():
    return {"status": "ok"}