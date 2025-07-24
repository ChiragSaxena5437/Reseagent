import os
import google.generativeai as genai
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


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
        for r in organic_results[:5] # Return top 5 results
    ]
    
    if not simplified_results:
        return "No good search results found."
        
    return json.dumps(simplified_results)


search_agent_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    tools=[search_the_web] 
)

def run_agent(agent, query):
    """
    Runs the agent and prints the conversation history to show the process.
    """
    print(f"\n--- Running Agent with query: '{query}' ---")
    
    chat = agent.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(query)

    print("\n--- Agent's Thought Process (Conversation History) ---")
    for content in chat.history:
        part_text = content.parts[0].text if content.parts[0].text else ""
        print(f"Role: {content.role}")
        print(f"Content: {part_text}")
        if (function_call := getattr(content.parts[0], "function_call", None)):
            print(f"Tool Call: {function_call.name}({function_call.args})")
        if (function_response := getattr(content.parts[0], "function_response", None)):
             print(f"Tool Response: {function_response.response}")

    print("\n--- Agent's Final Answer ---")
    print(response.text)


research_query = "What are the latest advancements and future challenges in using AI for drug discovery?"

run_agent(search_agent_model, research_query)