import operator
import json
import base64
import requests
from typing import TypedDict, List, Annotated, Literal, Optional, Dict
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================
# 1. PYDANTIC MODELS
# ==========================================
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=5, description="3-5 concrete, non-overlapping subpoints to cover in this section.")
    target_words: int = Field(..., description="Target word count for this section (120-450).")
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(
        ..., description="Use 'common_mistakes' and 'examples' exactly once in the plan."
    )

class Plan(BaseModel):
    blog_title: str = Field(..., description="A catchy and highly technical title for the blog post.")
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    tasks: List[Task]
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"

class RouterDecision(BaseModel):
    needs_research: bool = Field(description="Whether the topic requires external web search (True) or not (False).")
    reason: str = Field(description="Brief explanation of why research is or isn't needed.")
    queries: List[str] = Field(..., description="A list of 3 to 5 search queries. MUST NOT BE EMPTY if needs_research is true.")

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None 
    snippet: Optional[str] = None
    source: Optional[str] = None

class EvidenceWrapper(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="eg: [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    router_decision: RouterDecision
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[ImageSpec]
    verified_images: Dict[str, str]
    final: str

# ==========================================
# 3. LLM FACTORIES
# ==========================================
def get_gemini_llm():
    return ChatGoogleGenerativeAI(model='models/gemini-2.5-flash', temperature=0)

def get_ollama_llm():
    return ChatOllama(model="llama3.2", temperature=0)

def get_groq_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# ==========================================
# 4. TOOLS & HELPERS
# ==========================================
def _searxng_search(query: str, max_results: int = 5) -> List[dict]:
    url = "http://localhost:8080/search"
    params = {"q": query, "format": "json", "engines": "google,bing,duckduckgo", "language": "en-US"}
    try:
        response = requests.get(url=url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        normalized = []
        for r in data.get("results", [])[:max_results]:
            normalized.append({
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date"), 
                "source": r.get("engine"),
            })
        return normalized
    except Exception as e:
        logger.error(f"[SEARXNG] | Error calling SearxNG: {e}")
        return []

def _searxng_image_search(query: str, max_results: int = 3) -> List[dict]:
    url = "http://localhost:8080/search"
    params = {"q": query, "format": "json", "categories": "images", "engines": "google images,bing images"}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json().get("results", [])[:max_results]
    except:
        return []

def clean(text: str) -> str:
    if not text: return "---"
    return str(text).replace("|", "-").replace("\n", " ").strip()

@tool
def search_specific_site(query: str, site: str, max_results: int = 5) -> List[dict]:
    """
    Search for information within a specific website only.
    Use this when you have a high-quality source (like docs.python.org) 
    and need to find specific details within it.
    
    Args:
        query: The search term or question.
        site: The domain to search (e.g., 'wikipedia.org', 'github.com').
        max_results: How many results to return.
    """
    full_query = f"site:{site} {query}"
    results = _searxng_search(full_query, max_results=max_results)

    if not results:
        return "No results found for this site search."
            
    # Convert the list of dicts into a readable string for the LLM
    formatted_results = []
    for r in results:
        formatted_results.append(f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}")
        
    return "\n\n---\n\n".join(formatted_results)

# ==========================================
# 5. GRAPH NODES
# ==========================================
def router_node(state: State):
    topic = state["topic"]
    logger.info(f"[ROUTER] | Analyzing topic: {topic}")
    system_prompt = """You are an intelligent routing module for an AI research agent.\nYour objective is to analyze a user's topic and determine if external web research is required before generating a response.\n
    If needs_research=true:\n
    - Output 3-10 high-signal queries.
    - Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
    - If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES..
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Topic: {topic}")])
    chain = prompt | get_groq_llm().with_structured_output(RouterDecision) # Using Gemini for stable JSON
    result = chain.invoke({"topic": topic})
    
    logger.info(f"[ROUTER] | Decision -> {'Research Needed' if result.needs_research else 'Direct Orchestration'}")
    return {"router_decision": result.model_dump()}

def route_next(state: State) -> str:
    decision = RouterDecision(**state.get("router_decision", {}))
    return "research" if decision.needs_research else "orchestrator"

def research_node(state: State) -> dict:
    decision = state.get("router_decision", {})
    queries = decision.get("queries", [])
    logger.info(f"[RESEARCHER] | Gathering evidence for {len(queries)} queries")

    system_message = """You are a Specialized Research Extraction Agent. 

    Your goal is to transform raw, messy search engine results into a structured 'Evidence Pack'. 

    ### GUIDELINES:
    1. **Veracity First**: Only extract information that is explicitly stated in the snippets. Do not hallucinate details or "fill in the blanks."
    2. **Signal-to-Noise**: Ignore results that are clearly advertisements, cookie consent text, or irrelevant SEO filler.
    3. **Temporal Awareness**: If multiple results conflict, prioritize the one with the most recent 'published_at' date or the most authoritative 'source'.
    4. **Snippet Preservation**: Keep the 'snippet' meaningful. Do not summarize it so much that the specific data points (numbers, names, specs) are lost.
    5. **No Commentary**: Do not explain your choices. Simply output the filtered EvidenceItem list.

    ### OUTPUT EXPECTATION:
    You must return a valid EvidencePack. Each EvidenceItem must have a valid URL. If no relevant results are found, return an empty list."""

    raw_results = []
    for q in queries:
        logger.info(f"[RESEARCHER] | Searching for: {q}")
        raw_results.extend(_searxng_search(q, max_results=3))

    if not raw_results:
        logger.warning("[RESEARCHER] | No raw results found")
        return {"evidence": []}
    
    logger.info(f"[RESEARCHER] | Found {len(raw_results)} snippets. Extracting structured evidence...")
    extractor = get_gemini_llm().with_structured_output(EvidenceWrapper) # Using Gemini for stable JSON
    try:
        pack = extractor.invoke([
            ("system", system_message),
            ("human", f"Raw results:\n{json.dumps(raw_results, indent=2)}")
        ])
        dedup = {e.url: e for e in pack.evidence if e.url}
        logger.info(f"[RESEARCHER] | Research complete. {len(dedup)} unique items identified")
        return {"evidence": list(dedup.values())} 
    except Exception as e:
        logger.error(f"[RESEARCHER] | Extraction failed: {e}")
        return {"evidence": []}

def orchestrator_node(state: State) -> dict:
    logger.info(f"[ORCHESTRATOR] | Planning Blog Structure for topic: {state['topic']}")
    planner = get_ollama_llm().with_structured_output(Plan)
    
    orchestrator_system_message = """You are a Master Content Architect and Editor-in-Chief.
    Your goal is to create a detailed, high-signal execution plan for a technical blog post.

    ### YOUR INPUTS:
    1. **Topic**: The core subject requested by the user.
    2. **Scout Evidence**: Brief snippets from the web to ground your planning in current reality (e.g., current versions, recent news, existing controversies).

    ### YOUR MISSION:
    - **Determine Blog Kind**: Analyze the Topic and select the exact `blog_kind` that best fits the intent.
    - **Structure by Kind**: Tailor your logical task breakdown strictly to the chosen format:
        - *tutorial*: Step-by-step implementation, prerequisites, and code execution flow.
        - *system_design*: Architecture, component breakdowns, data flow, and system trade-offs.
        - *comparison*: Feature-by-feature analysis, pros/cons, and clear use-case recommendations.
        - *explainer*: Core concepts, mental models, and "how it works" under the hood.
        - *news_roundup*: Recent updates, release features, and industry impact analysis.
    - **Precision**: For each task, write a 'goal' and 'bullets' that are so specific that a worker with no context could write a perfect section.
    - **Research Guidance**: Provide 2-3 specific search queries for the workers to use for deep-diving into their specific section.
    - **Variety**: Ensure the 'section_type' distribution follows the requirements (using 'examples' and 'common_mistakes' exactly once).

    ### CRITICAL RULES:
    - If the 'Scout Evidence' mentions specific version numbers (e.g., Python 3.13) or dates, ensure the Plan reflects these.
    - Do not create more than 5-6 tasks total to ensure the blog remains focused.
    - Ensure the 'target_words' for the entire blog totals between 1000-1500 words."""
    
    evidence = state.get("evidence", [])
    evidence_str = "\n".join([f"[{i+1}] {e.title}\n Snippet: {e.snippet}\n URL: {e.url}\n" for i, e in enumerate(evidence[:10])]) if evidence else "No recent search evidence found."

    plan = planner.invoke([
        ("system", orchestrator_system_message),
        ("human", f"Topic: {state['topic']}\nScout Evidence:\n{evidence_str}")
    ])
    logger.info(f"[ORCHESTRATOR] | Plan generated: {plan.blog_title} with {len(plan.tasks)} tasks")
    return {"plan": plan.model_dump()}


def fanout(state: State):
    """
    This function reads the plan and spawns parallel workers.
    """
    plan = Plan(**state["plan"])
    if not plan or not plan.tasks:
        return []
    
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "plan": plan.model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])]
        })
        for task in plan.tasks
    ]

def worker(payload: dict) -> dict:
    plan = Plan(**payload["plan"])
    task = Task(**payload["task"])
    topic = payload.get("topic", "")
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    
    if evidence:
        rows = [f"| {clean(e.title)} | {e.url} | {clean(e.published_at)} | {clean(e.snippet)} |" for e in evidence[:5]]
        evidence_table = "\n".join(["| Title | URL | Date | Snippet |", "|-------|-----|------|---------|"] + rows)
    else:
        evidence_table = "No evidence available."

    system_prompt = f"""You are a professional Technical Writer. 
    Your goal is to write the '{task.section_type}' section of a blog post titled: "{plan.blog_title}".
    
    WRITING STYLE:
    - Audience: {plan.audience}
    - Tone: {plan.tone}
    - Blog Kind: {plan.blog_kind}
    - Standards: Use clean Markdown. Be precise, technical, and avoid fluff.

    CRITICAL OUTPUT RULES:
    1. You MUST output ONLY raw Markdown text. 
    2. DO NOT output JSON. DO NOT wrap your response in curly braces {{}}. 
    3. DO NOT output a function call. Just write the actual paragraphs and headers of the blog post.
    """
    
    human_message = f"""Please write the following section for our blog about '{topic}'.

    SECTION TITLE: {task.title}
    GOAL: {task.goal}
    
    KEY POINTS TO COVER:
    {bullets_text}

    RESEARCH EVIDENCE TO USE:
    {evidence_table}

    REQUIREMENTS:
    - Length: Approximately {task.target_words} words.
    - Format: Use Markdown headers (##) and bolding for key terms.
    - If there is a URL in the evidence that is highly relevant, cite it.
    """
    
    tools = [search_specific_site]
    llm = get_groq_llm().bind_tools(tools) # No tools bound here!
    try:
        response = llm.invoke([("system", system_prompt), ("human", human_message)])
        logger.info(f"[WORKER] | Finished section: {task.title}")
        return {"sections": [(task.id, response.content)]}
    except Exception as e:
        logger.error(f"[WORKER] | ❌ Worker failed for section '{task.title}': {e}")
        return {"sections": [(task.id, f"## {task.title}\n\n*Error: {str(e)}*")]}

def merge_node(state: State) -> dict:
    logger.info("[MERGE_NODE] | Merging the response from workers")
    raw_plan = state.get("plan", {})
    blog_title = raw_plan.get("blog_title", "Generated Blog") if isinstance(raw_plan, dict) else raw_plan.blog_title

    sorted_items = sorted(state.get("sections", []), key=lambda x: x[0])
    ordered_sections = []
    
    for item in sorted_items:
        md = item[1]
        if isinstance(md, list):
            md = md[0]["text"] if len(md) > 0 and isinstance(md[0], dict) and "text" in md[0] else str(md[0] if md else "")
        elif not isinstance(md, str):
            md = str(md)
        ordered_sections.append(md.strip())

    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {blog_title}\n\n{body}\n"
    return {"merged_md": merged_md, "final": merged_md}

def should_fetch_images(state: State) -> str:
    specs = state.get("image_specs", [])
    return "verify" if len(specs) > 0 else "skip"

def image_planner_node(state: State) -> dict:
    print("[IMAGE_PLANNER] | Planning whether images are needed")
    planner = get_gemini_llm().with_structured_output(GlobalImagePlan)
    raw_plan = state.get("plan", {})
    blog_kind = raw_plan.get("blog_kind", "explainer") if isinstance(raw_plan, dict) else getattr(raw_plan, "blog_kind", "explainer")

    system_message = f"""You are a Technical Visual Content Strategist. 
    Analyze the blog Markdown and identify where visual aids (diagrams, flowcharts, or technical photos) will add value.

    ACTION:
    1. Insert placeholders like [[IMAGE_1]], [[IMAGE_2]] directly into the text where they provide the most impact.
    2. For every placeholder, create a detailed ImageSpec.

    ### CRITICAL JSON OUTPUT RULES:
    1. You MUST output perfectly valid JSON. 
    2. The 'md_with_placeholders' field MUST contain the entire blog post with your placeholders inserted. Preserve all original markdown formatting using proper string escaping (\\n).
    3. You MUST include the 'images' array. If you decide no images are needed, return an empty array []. Do not forget this field!

    ### QUERY GENERATION RULES FOR SEARXNG:
    - The 'prompt' field MUST be a search-engine-optimized query.
    - Use technical keywords, not descriptive prose (e.g. "CRAG architecture diagram").
    - Ensure the image matches the 'blog_kind' ({blog_kind})."""
    
    try:
        image_plan = planner.invoke([
            ("system", system_message),
            ("human", f"Blog kind: {blog_kind}\nTopic: {state.get('topic', '')}\nBlog Content:\n{state.get('merged_md', '')}")
        ])
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.images] 
        }
    except Exception as e:
        print(f"[WARNING] Image Planner failed: {e}. Skipping images.")
        return {"md_with_placeholders": state.get('merged_md', ''), "image_specs": []}

def image_verification_node(state: State) -> dict:
    logger.info("[IMAGE_VERIFICATION] | Choosing proper images")
    llm = get_groq_llm()
    verified_images = {}
    
    raw_specs = state.get("image_specs", [])
    specs = []
    for item in raw_specs:
        if isinstance(item, dict): specs.append(ImageSpec(**item))
        elif isinstance(item, list): specs.extend([ImageSpec(**sub) for sub in item if isinstance(sub, dict)])
        elif isinstance(item, ImageSpec): specs.append(item)

    for spec in specs:
        logger.info(f"[IMAGE_VERIFICATION] | Searching image for: {spec.alt}")
        candidates = _searxng_image_search(spec.prompt)
        if not candidates: 
            logger.warning(f"[IMAGE_VERIFICATION] | No candidates found for placeholder: {spec.placeholder}")
            continue

        metadata = "\n".join([f"ID {i}: Title: {c.get('title')} | Source: {c.get('source', c.get('engine', 'Unknown'))}" for i, c in enumerate(candidates)])
        
        prompt = f"""Target Image: {spec.alt}
        {metadata}
        
        Which ID is the most relevant and high-quality technical image for this topic? 
        Return ONLY the number."""

        try:
            resp = llm.invoke(prompt).content.strip()
            idx = int(''.join(filter(str.isdigit, resp)))
            if 0 <= idx < len(candidates):
                best_url = candidates[idx].get('img_src', candidates[idx].get('url'))
                if best_url: 
                    verified_images[spec.placeholder] = best_url
                    logger.info(f"[IMAGE_VERIFICATION] | ✓ Selected index {idx} for {spec.placeholder}")
        except Exception as e:
            logger.warning(f"[IMAGE_VERIFICATION] | Error verifying {spec.placeholder}: {e}. Fallback to first.")
            if candidates:
                fallback = candidates[0].get('img_src', candidates[0].get('url'))
                if fallback: verified_images[spec.placeholder] = fallback

    return {"verified_images": verified_images}

def image_placement_node(state: State) -> dict:
    logger.info("[IMAGE_PLACEMENT] | Placing verified images into markdown")
    final_md = state.get("md_with_placeholders", "")
    verified_images = state.get("verified_images", {})
    
    raw_specs = state.get("image_specs", [])
    specs = []
    for item in raw_specs:
        if isinstance(item, dict): specs.append(ImageSpec(**item))
        elif isinstance(item, list): specs.extend([ImageSpec(**sub) for sub in item if isinstance(sub, dict)])
        elif isinstance(item, ImageSpec): specs.append(item)

    for spec in specs:
        if spec.placeholder in verified_images:
            url = verified_images[spec.placeholder]
            image_markdown = f"\n\n![{spec.alt}]({url})\n*Figure: {spec.caption}*\n\n"
            final_md = final_md.replace(spec.placeholder, image_markdown)
            logger.info(f"[IMAGE_PLACEMENT] | Placed image for {spec.placeholder}")
        else:
            final_md = final_md.replace(spec.placeholder, "")
            
    return {"final": final_md}

# ==========================================
# 6. GRAPH COMPILATION
# ==========================================
# Subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge", merge_node)
reducer_graph.add_node("image_planner", image_planner_node)
reducer_graph.add_node("image_verification", image_verification_node)
reducer_graph.add_node("image_placement", image_placement_node)

reducer_graph.add_edge(START, "merge")
reducer_graph.add_edge("merge", "image_planner")
reducer_graph.add_conditional_edges("image_planner", should_fetch_images, {"verify": "image_verification", "skip": END})
reducer_graph.add_edge("image_verification", "image_placement")
reducer_graph.add_edge("image_placement", END)
reducer_subgraph = reducer_graph.compile()

# Main Graph
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer_subgraph)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
graph.add_edge("research", "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

# Add Checkpointer Memory
# memory = MemorySaver()
# app_graph = graph.compile(checkpointer=memory)
workflow = graph