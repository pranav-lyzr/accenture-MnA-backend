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
import ast
from functools import wraps

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
        "agent-ID": "6817b49c2f31429a00a39d39",
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
        "agent-ID": "6818469e2f31429a00a39ddd",
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
        "agent-ID": "6818471a2f31429a00a39de1",
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
        "agent-ID": "681847532f31429a00a39de5",
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
        "agent-ID": "681847bb2f31429a00a39dec",
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
    }
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
        

# Lyzr agent query function
@async_retry(max_attempts=3, delay=2.0)
async def query_lyzr_agent(agent_id: str, session_id: str, prompt: str) -> Optional[Dict]:
    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": agent_id,
        "session_id": session_id,
        "message": "Return atleast 15-20 firms"
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
        print("RESPONSE",json_response)
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
        "agent_id": "68235606ffaabb77dd9b16fc",
        "session_id": "68235606ffaabb77dd9b16fc-bgio2t15bjl",
        "message": company_data_str
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "sk-default-PPcvzcCe4cJRRP8JkEXnT51woYJUXzMZ"
    }
    try:
        response = await async_client.post(
            "https://agent-dev.test.studio.lyzr.ai/v3/inference/chat/",
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
        "response": validated_companies,
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
@app.get("/prompts", summary="List Available Prompts", description="Returns the list of available search prompts.")
async def list_prompts() -> List[Dict]:
    return [{"index": i, "title": p["title"]} for i, p in enumerate(search_prompts)]

@app.post("/run_prompt", summary="Run a Specific Prompt", description="Runs a specific prompt by index and returns processed results. Optionally includes a custom message to refine the search.")
async def run_prompt(request: PromptRequest) -> Dict:
    try:
        prompt_data = search_prompts[request.prompt_index]
    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid prompt index")
    
    # Build the prompt content, incorporating the custom message if provided
    prompt_content = prompt_data['content']
    if request.custom_message:
        prompt_content += f"\n\nAdditional Search Requirements: {request.custom_message}"
    
    response = await query_perplexity(prompt_content, prompt_data['agent-ID'])
    if not response:
        raise HTTPException(status_code=500, detail="API request failed")

    processed = await process_api_response(response, prompt_data)
    
    # Save or update results in merger_search_results.json
    result_file = Path("merger_search_results.json")
    existing_results = {}
    if result_file.exists():
        with open(result_file, 'r') as f:
            existing_results = json.load(f)
    
    # Update results for this prompt
    existing_results[prompt_data['title']] = {
        "raw_response": processed.get("response", []),
        "extracted_companies": processed.get("companies", []),
        "validation_warnings": processed.get("validation_warnings", []),
        "custom_message": request.custom_message  # Save the custom message for reference
    }
    
    # Save updated results
    await save_to_json(existing_results)
    
    return processed

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

@app.post("/run_merger_search", summary="Run Full Merger Search", description="Executes the full merger search process, including running all prompts, extracting companies, saving results, and analyzing with Claude via Lyzr agent API, returning all responses as JSON objects.")
async def run_merger_search_endpoint() -> MergerSearchResponse:
    batch_tasks = [process_prompt_batch(p) for p in search_prompts]
    batch_results = await asyncio.gather(*batch_tasks)

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

    all_companies = sorted(list(set(all_companies)))
    io_tasks = [
        save_to_csv(all_companies),
        save_to_json(results)
    ]
    await asyncio.gather(*io_tasks)

    analysis_task = analyze_with_claude(all_companies, raw_responses)
    claude_analysis = await analysis_task

    return MergerSearchResponse(
        results={**results, "claude_analysis": claude_analysis},
        message="Merger search completed with parallel processing"
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

@app.get("/fetch_existing_items", summary="Fetch Existing Items", description="Retrieves the existing merger search results from the JSON file, if available.")
async def fetch_existing_items() -> Dict:
    result_file = Path("merger_search_results.json")
    if not result_file.exists():
        return {"results": {}, "message": "No existing results found"}
    with open(result_file, 'r') as f:
        results = json.load(f)
    return {"results": results, "message": "Successfully retrieved existing results"}

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