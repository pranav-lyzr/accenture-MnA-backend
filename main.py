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

def sanitize_json_string(json_str: str) -> str:
    """
    Sanitizes a JSON string by removing code block wrappers, inline comments,
    and extracting embedded JSON from narrative text. Returns valid JSON or error object.
    """
    if not json_str or json_str.isspace():
        logger.warning("Empty or whitespace JSON string received")
        return "{}"

    # Remove non-printable characters and normalize whitespace
    json_str = "".join(c for c in json_str if c.isprintable()).strip()

    # Remove code block wrappers and inline comments
    json_str = json_str.replace('```json', '').replace('```', '').replace('//', '')

    # Try parsing the entire string
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}. Raw response (first 200 chars): {json_str[:200]}...")

        # Extract embedded JSON from narrative text
        json_match = re.search(r'\{[\s\S]*\}', json_str)
        if json_match:
            json_str = json_match.group(0)
            try:
                json.loads(json_str)
                logger.info("Extracted embedded JSON successfully.")
                return json_str
            except json.JSONDecodeError:
                pass

        # Fix common JSON issues: trailing commas, missing commas
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)  # Remove trailing commas
        json_str = re.sub(r'}(\s*[{[])', r'},\1', json_str)  # Add missing commas

        # Try parsing again
        try:
            json.loads(json_str)
            logger.info("Sanitization successful.")
            return json_str
        except json.JSONDecodeError:
            logger.error("Sanitization failed. Wrapping as error message.")
            return json.dumps({"error": "Invalid JSON response", "raw_content": json_str})        

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

# Search prompts (unchanged, as provided by user)
search_prompts = [
    {
        "title": "Initial Target Identification",
        "content": """Identify 5-7 boutique retail consulting firms in North America as potential merger candidates:

Criteria:
- Annual revenue: $10-20M
- Specialized expertise: Product development and sourcing strategy
- Primary client base: Enterprise retailers (revenue >$500M)
- Headquartered in North America

Research Methodology:
- Cross-reference data from company websites, industry reports (e.g., IBISWorld, Consultancy.org), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, press releases, and client testimonials
- Compare firms against reference models like 703 Advisors, Alix Partners, and OC&C for strategic fit

Comprehensive Evaluation Dimensions:
- Financial Performance: Revenue, growth rate (past 3-5 years), profitability (e.g., EBITDA margins), valuation multiples
- Operational Scale: Employee count, office locations, global delivery capabilities
- Client Portfolio: Key enterprise retail clients, average contract value, notable project outcomes
- Leadership: Key executives, their industry experience, thought leadership
- Specialization: Depth in product development and sourcing strategy
- Merger Fit: Strategic synergies, cultural alignment, integration challenges
- Technology: Tools and platforms used for client solutions

Output Requirements:
- Provide a JSON object for each firm with the following structure:
  ```json
  {
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "profitability": "<string, e.g., 20% EBITDA margin>",
    "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "office_locations": ["<string>", ...],
    "key_clients": ["<string>", ...],
    "average_contract_value": "<string, e.g., $500K>",
    "leadership": [
      {
        "name": "<string>",
        "title": "<string>",
        "experience": "<string, e.g., 15 years in retail consulting>"
      },
      ...
    ],
    "primary_domains": ["<string>", ...],
    "proprietary_methodologies": "<string, e.g., Data-driven sourcing model>",
    "technology_tools": ["<string>", ...],
    "competitive_advantage": "<string>",
    "merger_synergies": "<string>",
    "cultural_alignment": "<string>",
    "integration_challenges": "<string>",
    "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
    "sources": ["<string>", ...]
  }
  ```
- Include 2-3 credible sources per firm (e.g., company website, LinkedIn, industry reports)
- If data is unavailable, provide estimates with assumptions or mark as "Data not available"
- Ensure results are concise, actionable, and suitable for executive evaluation of merger candidates.""",
        "sources": [
            "https://www.703advisors.com/",
            "https://www.alixpartners.com/",
            "https://www.occstrategy.com/",
            "https://www.consultancy.uk/consultancy-size/revenue",
            "https://www.consultancy.org/consulting-industry/firm-size",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/",
            "https://www.ibisworld.com/"
        ]
    },
    {
        "title": "Market Analysis and Competitive Landscape",
        "content": """Conduct a comprehensive competitive analysis of mid-tier retail strategy consulting firms to identify top merger candidates:

Criteria:
- Headquartered in North America
- Annual revenue: $10-20M
- Core competencies: Product Development, Sourcing Strategy, Procurement Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Analyze at least 10 firms
- Cross-reference data from company websites, industry reports (e.g., IBISWorld, Gartner), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, press releases, and client testimonials
- Exclude firms with incomplete or unverified data

Comprehensive Evaluation Dimensions:
- Financial Performance: Estimated annual revenue, revenue growth rate (past 3-5 years), profitability (e.g., EBITDA margins), valuation multiples
- Operational Scale: Number of employees, office locations, global delivery capabilities
- Client Portfolio: Key enterprise retail clients, average contract value, notable case studies
- Leadership: Key executives, their industry experience, thought leadership
- Specialization Depth: Expertise in product development, sourcing strategy, procurement optimization
- Technology Integration: Use of tools/platforms (e.g., AI, analytics, supply chain software)
- Merger Fit: Strategic synergies, cultural alignment, integration challenges
- Innovation: Proprietary methodologies, patents, or unique processes
- Market Positioning: Market penetration in enterprise retail consulting

Output Requirements:
- Provide a JSON object for each firm with the following structure:
  ```json
  {
    "rank": <integer>,
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "profitability": "<string, e.g., 20% EBITDA margin>",
    "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "office_locations": ["<string>", ...],
    "key_clients": ["<string>", ...],
    "average_contract_value": "<string, e.g., $500K>",
    "leadership": [
      {
        "name": "<string>",
        "title": "<string>",
        "experience": "<string, e.g., 15 years in retail consulting>"
      },
      ...
    ],
    "primary_domains": ["<string>", ...],
    "proprietary_methodologies": "<string, e.g., Analytics-driven sourcing framework>",
    "technology_tools": ["<string>", ...],
    "competitive_advantage": "<string>",
    "merger_synergies": "<string>",
    "cultural_alignment": "<string>",
    "integration_challenges": "<string>",
    "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
    "sources": ["<string>", ...]
  }
  ```
- Rank firms from 1 to 10 based on merger potential (financial stability, strategic fit, market positioning)
- Include 2-3 credible sources per firm
- If data is unavailable, provide estimates with assumptions or mark as "Data not available"
- Ensure results are concise and suitable for executive merger evaluations.""",
        "sources": [
            "https://www.consultancy.org/consulting-industry/market-analysis",
            "https://www.ibisworld.com/industry-trends/market-research-reports/",
            "https://www.gartner.com/en/consulting/insights/market-insights",
            "https://www.forrester.com/bold",
            "https://www.mckinsey.com/industries/retail/our-insights",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/",
            "https://www.bloomberg.com/markets"
        ]
    },
    {
        "title": "Sourcing and Procurement Specialists",
        "content": """Conduct a comprehensive survey of boutique procurement strategy consulting firms for merger potential:

Target Firms:
- Headquartered in North America
- Annual revenue: $10-20M
- Specializations: Strategic Sourcing, Global Procurement, Retail Supply Chain Optimization
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Analyze at least 7 firms
- Cross-reference data from company websites, industry reports (e.g., Procurement Leaders, CIPS), financial databases (e.g., PitchBook), and LinkedIn
- Validate through public records, client testimonials, and case studies
- Exclude firms with incomplete or unverified data

Evaluation Criteria:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations, global sourcing reach
- Client Portfolio: Key retail clients, average contract value, notable sourcing projects
- Leadership: Key executives, their procurement expertise, thought leadership
- Sourcing Complexity: Handling of regional vs. global sourcing challenges
- Technology: Tools for procurement optimization (e.g., SAP Ariba, Coupa)
- Merger Fit: Synergies, cultural alignment, integration challenges
- Emerging Markets: Expertise in sourcing from emerging markets

Output Requirements:
- Provide a JSON object for each firm with the following structure:
  ```json
  {
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "profitability": "<string, e.g., 20% EBITDA margin>",
    "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "office_locations": ["<string>", ...],
    "key_clients": ["<string>", ...],
    "average_contract_value": "<string, e.g., $500K>",
    "leadership": [
      {
        "name": "<string>",
        "title": "<string>",
        "experience": "<string, e.g., 15 years in procurement>"
      },
      ...
    ],
    "primary_domains": ["<string>", ...],
    "proprietary_methodologies": "<string, e.g., Predictive sourcing analytics>",
    "technology_tools": ["<string>", ...],
    "competitive_advantage": "<string>",
    "merger_synergies": "<string>",
    "cultural_alignment": "<string>",
    "integration_challenges": "<string>",
    "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
    "technological_enablement_score": "<string, e.g., High, based on AI adoption>",
    "global_sourcing_reach": "<string, e.g., Strong in Asia and LATAM>",
    "sources": ["<string>", ...]
  }
  ```
- Include 2-3 credible sources per firm
- If data is unavailable, provide estimates with assumptions or mark as "Data not available"
- Ensure results are actionable for executive merger evaluations.""",
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
        "content": """Map the retail product development consulting ecosystem to identify merger candidates:

Firm Selection Criteria:
- Headquartered in North America
- Annual revenue: $10-20M
- Expertise in: Concept Development, Product Optimization, Launch Strategy, Post-Launch Performance Analysis
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Analyze at least 7 firms
- Cross-reference data from company websites, industry reports (e.g., PDMA, Retail Dive), financial databases (e.g., PitchBook), and LinkedIn
- Validate through case studies, client testimonials, and public records
- Exclude firms with incomplete or unverified data

Comprehensive Assessment Dimensions:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations, global capabilities
- Client Portfolio: Key retail clients, average contract value, representative case studies
- Leadership: Key executives, their product development expertise, thought leadership
- Innovation: Success rate, time-to-market acceleration, cross-category capabilities
- Technology: Tools for product development (e.g., PLM software, analytics)
- Merger Fit: Synergies, cultural alignment, integration challenges
- Retail Specialization: Depth in retail verticals (e.g., apparel, electronics)

Output Requirements:
- Provide a JSON object for each firm with the following structure:
  ```json
  {
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "profitability": "<string, e.g., 20% EBITDA margin>",
    "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "office_locations": ["<string>", ...],
    "key_clients": ["<string>", ...],
    "average_contract_value": "<string, e.g., $500K>",
    "leadership": [
      {
        "name": "<string>",
        "title": "<string>",
        "experience": "<string, e.g., 15 years in product development>"
      },
      ...
    ],
    "primary_domains": ["<string>", ...],
    "proprietary_methodologies": "<string, e.g., Agile product optimization>",
    "technology_tools": ["<string>", ...],
    "competitive_advantage": "<string>",
    "merger_synergies": "<string>",
    "cultural_alignment": "<string>",
    "integration_challenges": "<string>",
    "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
    "innovation_metrics": "<string, e.g., 80% success rate in product launches>",
    "case_studies": ["<string>", ...],
    "sources": ["<string>", ...]
  }
  ```
- Include 2-3 credible sources per firm
- If data is unavailable, provide estimates with assumptions or mark as "Data not available"
- Ensure results are actionable for executive merger evaluations.""",
        "sources": [
            "https://www.pdma.org/page/ResearchReports",
            "https://www.innovationmanagement.se/",
            "https://www.fastcompany.com/innovation-insights",
            "https://www.retaildive.com/news/product-development/",
            "https://www.productstrategygroup.com/resources",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/"
        ]
    },
    {
        "title": "Supply Chain Specialists",
        "content": """Map the supply chain consulting ecosystem to identify strategic optimization specialists for merger potential:

Firm Selection Parameters:
- Headquartered in North America
- Annual revenue: $10-20M
- Core Competencies: Supply Chain Optimization, Vendor Assessment/Evaluation, Digital Supply Chain Transformation
- Primary client base: Enterprise retailers (revenue >$500M)

Research Methodology:
- Analyze at least 7 firms
- Cross-reference data from company websites, industry reports (e.g., Gartner, Supply Chain Dive), financial databases (e.g., PitchBook), and LinkedIn
- Validate through case studies, client testimonials, and public records
- Exclude firms with incomplete or unverified data

Comprehensive Evaluation Framework:
- Financial Performance: Revenue, growth rate, profitability, valuation multiples
- Operational Scale: Employee count, office locations, global capabilities
- Client Portfolio: Key retail clients, average contract value, notable projects
- Leadership: Key executives, their supply chain expertise, thought leadership
- Supply Chain Complexity: Handling of complex supply chains (e.g., multi-region)
- Technology: Tools for supply chain optimization (e.g., Blue Yonder, Kinaxis)
- Merger Fit: Synergies, cultural alignment, integration challenges
- Sustainability: Integration of sustainable practices in supply chain solutions

Output Requirements:
- Provide a JSON object for each firm with the following structure:
  ```json
  {
    "name": "<string>",
    "domain_name": "<string>",
    "estimated_revenue": "<string, e.g., $15M>",
    "revenue_growth": "<string, e.g., 10% CAGR over 3 years>",
    "profitability": "<string, e.g., 20% EBITDA margin>",
    "valuation_estimate": "<string, e.g., $30M based on 2x revenue multiple>",
    "employee_count": "<string, e.g., 50-75 employees>",
    "office_locations": ["<string>", ...],
    "key_clients": ["<string>", ...],
    "average_contract_value": "<string, e.g., $500K>",
    "leadership": [
      {
        "name": "<string>",
        "title": "<string>",
        "experience": "<string, e.g., 15 years in supply chain>"
      },
      ...
    ],
    "primary_domains": ["<string>", ...],
    "proprietary_methodologies": "<string, e.g., Digital supply chain framework>",
    "technology_tools": ["<string>", ...],
    "competitive_advantage": "<string>",
    "merger_synergies": "<string>",
    "cultural_alignment": "<string>",
    "integration_challenges": "<string>",
    "market_penetration": "<string, e.g., 5% of enterprise retail consulting market>",
    "digital_transformation_score": "<string, e.g., High, based on AI adoption>",
    "sustainability_performance": "<string, e.g., 30% reduction in carbon footprint>",
    "sources": ["<string>", ...]
  }
  ```
- Include 2-3 credible sources per firm
- If data is unavailable, provide estimates with assumptions or mark as "Data not available"
- Ensure results are actionable for executive merger evaluations.""",
        "sources": [
            "https://www.supplychaindigital.com/technology/top-supply-chain-consulting-firms",
            "https://www.gartner.com/en/supply-chain/insights",
            "https://www.supplychaindive.com/",
            "https://www.mckinsey.com/capabilities/operations/our-insights/supply-chain-insights",
            "https://www.consultancy.uk/sectors/supply-chain-consulting",
            "https://www.pitchbook.com/",
            "https://www.linkedin.com/"
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

# Apollo API enrichment function (unchanged)
def enrich_company(company_domain: Optional[str] = None, log_file: Optional[str] = "apollo_enrich.log") -> Dict:
    url = "https://api.apollo.io/api/v1/organizations/enrich"
    
    params = {}
    if company_domain and isinstance(company_domain, str) and company_domain.strip():
        params["domain"] = company_domain
    else:
        return {"error": "Company domain is required for enrichment"}
    
    headers = {
        "accept": "application/json",
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "x-api-key": APOLLO_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get("organization"):
            return data["organization"]
        else:
            return {"error": "Company not found or insufficient matching data", "response": data}
    
    except requests.exceptions.RequestException as e:
        error_detail = {"error": f"API request failed: {str(e)}"}
        if isinstance(e, requests.exceptions.HTTPError) and e.response:
            try:
                error_detail["response"] = e.response.json()
            except ValueError:
                error_detail["response"] = e.response.text
        logger.error(f"Apollo API error: {error_detail}")
        return error_detail

# API Functions
@retry_api_call(max_attempts=3, delay=2.0)
async def query_lyzr_agent(agent_id: str, session_id: str, prompt: str) -> Optional[Dict]:
    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": agent_id,
        "session_id": session_id,
        "message": prompt
    }
    try:
        response = requests.post(LYZR_AGENT_URL, json=payload, headers=LYZR_AGENT_HEADERS)
        response.raise_for_status()
        json_response = response.json()
        raw_response = json_response.get("response", "{}")
        logger.info(f"Raw API response (first 200 chars): {raw_response}...")
        logger.info(f"API response size: {len(str(json_response))} bytes")
        json_response["response"] = sanitize_json_string(raw_response)
        return json_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Lyzr agent API error: {e}")
        raise

async def query_perplexity(prompt: str) -> Optional[Dict]:
    return await query_lyzr_agent(LYZR_PERPLEXITY_AGENT_ID, LYZR_PERPLEXITY_SESSION_ID, prompt)

async def query_claude(prompt: str) -> Optional[Dict]:
    return await query_lyzr_agent(LYZR_CLAUDE_AGENT_ID, LYZR_CLAUDE_SESSION_ID, prompt)

def extract_companies(json_response: Dict) -> List[str]:
    try:
        companies = json_response.get("companies", [])
        return [company.get("name", "") for company in companies if company.get("name")]
    except (AttributeError, TypeError) as e:
        logger.warning(f"Invalid JSON structure for companies extraction: {e}")
        return []

async def analyze_with_claude(companies: List[str], raw_responses: Dict[str, str]) -> Optional[Dict]:
    full_context = "\n\n".join([f"### {title}\n{content}" for title, content in raw_responses.items()])
    company_list = "\n".join([f"- {company}" for company in companies])
    analysis_prompt = """Analyze potential merger candidates for a retail consulting company in the USA with the following criteria:
- Size: ~$15M in sales
- Industry: Retail consulting
- Client Type: Large enterprise retailers (>$500M revenue)
- Key Services: Product development and sourcing strategy

Based on my research, I've identified the following companies:

{company_list}

Here is the detailed information I've collected about these companies:

{full_context}

Please provide a comprehensive analysis of these companies as potential merger candidates, tailored for presentation to senior executives. Use the provided data fields (e.g., estimated_revenue, revenue_growth, profitability, employee_count, key_clients, leadership, proprietary_methodologies, technology_tools, merger_synergies, cultural_alignment, integration_challenges) to inform your analysis.

Evaluation Framework:
1. Financial Health (30%): Assess estimated_revenue, revenue_growth, profitability, and valuation_estimate. Prioritize firms with stable growth (e.g., >8% CAGR) and strong margins (e.g., >15% EBITDA).
2. Strategic Fit (30%): Evaluate alignment in primary_domains, key_clients, and proprietary_methodologies with the acquiring firm's focus on product development and sourcing strategy.
3. Operational Compatibility (20%): Consider employee_count, office_locations, and global capabilities to ensure scalable integration.
4. Leadership and Innovation (10%): Analyze leadership expertise and technology_tools to assess long-term value creation.
5. Cultural and Integration Risks (10%): Review cultural_alignment and integration_challenges to minimize post-merger friction.

Analysis Requirements:
1. Rank the top 5 most promising merger candidates based on the evaluation framework. Provide a clear explanation for each ranking, referencing specific data fields.
2. For each top candidate, provide:
   - Estimated Valuation: Calculate using provided estimated_revenue and profitability, applying industry-standard multiples (e.g., 1.5-3x revenue or 8-12x EBITDA for boutique retail consulting). Validate against valuation_estimate if provided.
   - Strategic Fit Assessment: Detail complementarity in services (e.g., product development vs. sourcing), client base overlap, and market expansion potential.
   - Potential Synergies: Identify specific benefits (e.g., cross-selling to key_clients, combining proprietary_methodologies, leveraging technology_tools).
   - Integration Challenges: Highlight risks (e.g., cultural misalignment, service overlap, geographic constraints) and mitigation strategies.
   - References: Cite specific sources from the input data (e.g., company website, LinkedIn, industry reports) and additional credible sources if needed.
3. Recommend the single best merger candidate with a detailed justification, summarizing why it excels in financial health, strategic fit, and operational compatibility, and how it minimizes integration risks.

Output Requirements:
- Provide a JSON object with the following structure:
  ```json
  {
    "analysis": {
      "top_candidates": [
        {
          "rank": <integer>,
          "name": "<string>",
          "domain_name": "<string>",
          "reason": "<string, e.g., Strong financials and complementary services>",
          "valuation": "<string, e.g., $30M based on 2x revenue>",
          "strategic_fit": "<string, e.g., Aligns with sourcing strategy>",
          "synergies": "<string, e.g., Cross-selling to shared clients>",
          "challenges": "<string, e.g., Cultural integration risks>",
          "references": ["<string, e.g., https://company.com>", ...]
        },
        ...
      ],
      "recommended_candidate": {
        "name": "<string>",
        "justification": "<string, e.g., Best balance of financial stability and strategic fit>"
      }
    }
  }
  ```
- Ensure the analysis is concise, data-driven, and suitable for a boardroom presentation.
- Use provided data fields explicitly (e.g., 'Based on {company.estimated_revenue} and {company.revenue_growth}...').
- If data is missing, make reasonable assumptions based on industry norms (e.g., 15-20% EBITDA margins for boutique firms) and note them.
- Cross-reference sources from {full_context} and add external sources (e.g., Consultancy.org, PitchBook) for validation where applicable.

Ensure the analysis reflects deep industry knowledge, strategic foresight, and practical merger considerations to impress a seasoned retail consulting executive."""
    response = await query_claude(analysis_prompt)
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

@app.post("/run_prompt", summary="Run a Single Search Prompt", description="Executes a single search prompt using the Lyzr agent API for Perplexity, expecting JSON output, and returns the response as a JSON object and extracted companies.")
async def run_prompt(request: PromptRequest) -> Dict:
    if request.prompt_index >= len(search_prompts):
        raise HTTPException(status_code=400, detail="Invalid prompt index")
    
    prompt_data = search_prompts[request.prompt_index]
    response = await query_perplexity(prompt_data['content'])
    if not response:
        raise HTTPException(status_code=500, detail="Lyzr agent API request failed")
    
    try:
        json_content = json.loads(sanitize_json_string(response.get('response', '{}')))
        companies = extract_companies(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response as JSON: {e}")
        json_content = {"error": "Invalid JSON response", "raw_content": response.get('response', '')}
        companies = []

    return {
        "title": prompt_data["title"],
        "raw_response": json_content,
        "extracted_companies": companies,
        "sources": prompt_data["sources"]
    }

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
    results: Dict = {}
    all_companies: List[str] = []
    raw_responses: Dict[str, str] = {}

    # Step 1: Run all search prompts
    for i, prompt_data in enumerate(search_prompts):
        logger.info(f"Running prompt {i+1}/{len(search_prompts)}: {prompt_data['title']}")
        response = await query_perplexity(prompt_data['content'])
        if response:
            try:
                json_content = json.loads(sanitize_json_string(response.get('response', '{}')))
                companies = extract_companies(json_content)
                raw_responses[prompt_data['title']] = json.dumps(json_content, indent=2)
                all_companies.extend(companies)
                results[prompt_data['title']] = {
                    "raw_response": json_content,
                    "extracted_companies": companies
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response for {prompt_data['title']} as JSON: {e}")
                json_content = {"error": "Invalid JSON response", "raw_content": response.get('response', '')}
                results[prompt_data['title']] = {
                    "raw_response": json_content,
                    "extracted_companies": []
                }
        else:
            logger.error(f"Lyzr agent API request failed for {prompt_data['title']}")
            results[prompt_data['title']] = {
                "raw_response": {"error": "API request failed"},
                "extracted_companies": []
            }
        time.sleep(2)  # Avoid rate limiting

    # Step 2: Consolidate companies
    all_companies = sorted(list(set(all_companies)))
    results["consolidated_companies"] = all_companies

    # Step 3: Save to CSV
    companies_df = pd.DataFrame({"Company": all_companies})
    companies_df.to_csv("identified_companies.csv", index=False)

    # Step 4: Claude analysis
    claude_analysis = await analyze_with_claude(all_companies, raw_responses)
    results["claude_analysis"] = claude_analysis

    # Step 5: Save to JSON
    with open('merger_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return MergerSearchResponse(
        results=results,
        message="Merger search completed. Results saved to 'identified_companies.csv' and 'merger_search_results.json'."
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
    result = enrich_company(company_domain=request.company_domain)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
