import os
import google.generativeai as genai
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json
import requests
from bs4 import BeautifulSoup

# --- 1. Environment and API Key Setup ---
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# --- 2. Tool Definitions ---
def search_the_web(query: str):
    """Performs a web search for the given query."""
    print(f"--- TOOL CALLED: search_the_web('{query}') ---")
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google"}
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    simplified_results = [{"link": r.get("link")} for r in organic_results[:3]]
    return json.dumps(simplified_results, indent=2)

def scrape_website_content(url: str):
    """Scrapes the main text content from a URL."""
    print(f"--- TOOL CALLED: scrape_website_content('{url}') ---")
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        main_content = ' '.join([p.get_text() for p in paragraphs])
        if not main_content: return "Error: No text content found on the page."
        return main_content[:8000]
    except requests.RequestException as e:
        return f"Error scraping website: {e}"

def write_the_report(researched_content: str):
    """Generates a polished report from researched content."""
    print(f"--- TOOL CALLED: write_the_report ---")
    writer_agent_model = genai.GenerativeModel("gemini-1.5-flash-latest",
        system_instruction="""You are a professional report writer. Your task is to take the provided researched content and transform it into a high-quality, well-structured report. If the content is insufficient, state that a report cannot be generated.""")
    response = writer_agent_model.generate_content(researched_content)
    return response.text

TOOL_REGISTRY = {
    "search_the_web": search_the_web,
    "scrape_website_content": scrape_website_content,
    "write_the_report": write_the_report,
}

# --- 3. Agent Definition (The Manager) ---
manager_agent_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    tools=[search_the_web, scrape_website_content, write_the_report],
    system_instruction="""
    You are a research project manager. You orchestrate a team of specialist agents to produce a final report.

    Your workflow is strict:
    1.  Use the `search_the_web` tool to find relevant sources for the user's query.
    2.  Use the `scrape_website_content` tool on the top 2 links to gather the raw information. You can do this in parallel.
    3.  Compile all the successfully scraped content into a single block of text.
    4.  Finally, you MUST use the `write_the_report` tool, passing all your researched content to it.
    
    Your final output to the user MUST be the direct result from the `write_the_report` tool.
    """
)


# --- 4. Execution (Manual Control Loop for Parallel Calls) ---
def run_manager(agent, query):
    """Runs the manager agent with a manual loop that handles parallel tool calls."""
    print(f"\n--- Running Manager with query: '{query}' ---")
    
    chat_session = agent.start_chat()
    # Initial message to kick off the process
    response = chat_session.send_message(query)

    # Loop until the model gives a text response
    while True:
        # Check if the model's response contains tool calls
        if not response.candidates[0].content.parts or not response.candidates[0].content.parts[0].function_call:
            # If no tool calls, it's the final answer
            print("\n--- Final Report from the Writer Agent ---")
            print(response.text)
            break

        # If there are tool calls, process them
        tool_responses = []
        for part in response.candidates[0].content.parts:
            function_call = part.function_call
            tool_name = function_call.name
            tool_args = {key: value for key, value in function_call.args.items()}
            
            # Look up the function and execute it
            tool_function = TOOL_REGISTRY[tool_name]
            tool_output = tool_function(**tool_args)
            
            # Collect the response for each tool call
            tool_responses.append({
                "function_response": {
                    "name": tool_name,
                    "response": {"content": tool_output}
                }
            })
        
        # Send all tool responses back to the model in a single message
        response = chat_session.send_message(tool_responses)


# --- Define the research question ---
research_query = "What are the latest advancements and future challenges in using AI for drug discovery?"

# --- Run the manager ---
run_manager(manager_agent_model, research_query)