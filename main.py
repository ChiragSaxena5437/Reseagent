import os
import google.generativeai as genai
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json

# --- 1. Environment and API Key Setup ---
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# --- 2. Tool Definition ---
def search_the_web(query: str):
    """
    Performs a web search using the provided query and returns the top 5 organic results.
    The result is a JSON string containing a list of dictionaries with 'title', 'link', and 'snippet'.
    """
    print(f"--- TOOL CALLED: Searching the web for '{query}' ---")
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    
    simplified_results = [
        {"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")} 
        for r in organic_results[:5]
    ]
    
    if not simplified_results:
        return "No good search results found."
        
    return json.dumps(simplified_results, indent=2)


# --- 3. Agent Definition ---
# This is the corrected agent definition with explicit instructions.
search_agent_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    # This system instruction is the key to fixing the problem.
    # It forces the model to use the tool.
    system_instruction="""
    You are a helpful research assistant. Your purpose is to answer user questions.
    You MUST use the `search_the_web` tool to answer the user's query. Do not rely on your internal knowledge.
    After getting the results from the tool, present the raw JSON results to the user.
    """,
    tools=[search_the_web]
)


# --- 4. Execution ---
def run_agent(agent, query):
    """
    Runs the agent and prints the final response.
    """
    print(f"\n--- Running Agent with query: '{query}' ---")
    
    chat = agent.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(query)

    print("\n--- Agent's Final Answer ---")
    print(response.text)


# Define the research question
research_query = "What are the latest advancements and future challenges in using AI for drug discovery?"

# Run the agent
run_agent(search_agent_model, research_query)