import os
import google.generativeai as genai
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json
import requests
from bs4 import BeautifulSoup

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def search_the_web(query: str):
    """
    Performs a web search using the provided query and returns the top 3 organic results.
    The result is a JSON string containing a list of dictionaries with 'title', 'link', and 'snippet'.
    """
    print(f"--- TOOL CALLED: Searching the web for '{query}' ---")
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google"}
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    
    simplified_results = [
        {"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")} 
        for r in organic_results[:3] # Return top 3 results
    ]
    
    if not simplified_results:
        return "No good search results found."
        
    return json.dumps(simplified_results, indent=2)

def scrape_website_content(url: str):
    """
    Scrapes the main text content from a given URL. This is used to get the content for summarization.
    """
    print(f"--- TOOL CALLED: Scraping website {url} ---")
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        main_content = ' '.join([p.get_text() for p in paragraphs])
        
        return main_content[:8000]

    except requests.RequestException as e:
        return f"Error scraping website: {e}"


manager_agent_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    tools=[search_the_web, scrape_website_content],
    system_instruction="""
    You are a highly skilled research manager. Your job is to produce a detailed and comprehensive report on a user's topic.

    Your workflow is as follows:
    1.  First, you MUST use the `search_the_web` tool with a clear and effective query to find relevant articles.
    2.  Next, for the top 2 most promising URLs from the search results, you MUST use the `scrape_website_content` tool to extract the text from each of those websites. Do not scrape more than 2.
    3.  After you have the content from both websites, you MUST synthesize this information into a single, cohesive, well-written report.
    4.  The final report should have a clear introduction, a body that discusses the key findings from the sources, and a concluding summary.
    5.  Do NOT present the raw scraped text to the user. Your final output must be the polished report.
    """
)


def run_manager(agent, query):
    """
    Runs the manager agent and prints the final, synthesized report.
    """
    print(f"\n--- Running Manager with query: '{query}' ---")
    
    chat = agent.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(query)

    print("\n--- Manager's Final Report ---")
    print(response.text)


try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Please install required libraries: pip install requests beautifulsoup4")
    exit()


research_query = "What are the latest advancements and future challenges in using AI for drug discovery?"

run_manager(manager_agent_model, research_query)