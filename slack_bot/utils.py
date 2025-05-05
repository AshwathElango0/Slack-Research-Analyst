import os
import re
import pickle
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import ArxivLoader
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from duckduckgo_search import DDGS

def extract_arxiv_id(entry_id: str) -> str:
    match = re.search(r'arxiv\.org/abs/([^/]+)', entry_id)
    return match.group(1) if match else entry_id

def extract_keywords_from_query(query: str) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    # Arbitrarily adding some common stopwords which we can filter out
    stopwords = {"what", "when", "where", "which", "with", "have", "this", "that", "about", "from", "your", "query", "could", "would", "should"}
    return list(set(w for w in words if w not in stopwords))

# ---------- Constants and Models ----------
MEMORY_DIR = "user_memories"
FAISS_PATH = "faiss_store.pkl"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
user_memories: Dict[str, BaseChatMemory] = {}
user_preferences: Dict[str, Dict[str, Any]] = {}
user_profiles: Dict[str, Dict[str, Any]] = {}

def search_duckduckgo(query: str, max_results: int = 3) -> str:
    print(f"[WEB SEARCH] Searching DuckDuckGo for: {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title")
            href = r.get("href")
            snippet = r.get("body")
            results.append(f"- [{title}]({href}): {snippet}")
    return "\n".join(results) if results else "No useful results found."

# ---------- Preferences ----------
def get_contextual_profile(user_id: str) -> Dict[str, Any]:
    if user_id not in user_profiles:
        user_profiles[user_id] = {
            "topics": [],
            "technical_level": "unknown",  # beginner/intermediate/advanced are some possible options
            "question_types": {},
            "last_queries": [],
        }
    return user_profiles[user_id]

def update_contextual_profile(user_id: str, query: str, intent: str, keywords: List[str]):
    profile = get_contextual_profile(user_id)

    # Track query types
    profile["question_types"][intent] = profile["question_types"].get(intent, 0) + 1

    # Track common keywords
    profile["topics"].extend(k for k in keywords if k not in profile["topics"])
    profile["topics"] = list(set(profile["topics"]))

    # Track recent queries
    profile["last_queries"].append(query)
    if len(profile["last_queries"]) > 10:
        profile["last_queries"] = profile["last_queries"][-10:]

def get_user_preferences(user_id: str) -> Dict[str, Any]:
    if user_id not in user_preferences:
        user_preferences[user_id] = {"interests": [], "focus": ""}
    return user_preferences[user_id]

def update_user_preferences(user_id: str, new_prefs: Dict[str, Any]):
    prefs = get_user_preferences(user_id)
    for key, value in new_prefs.items():
        if isinstance(prefs.get(key), list) and isinstance(value, list):
            prefs[key] = list(set(prefs[key] + value))
        else:
            prefs[key] = value
    user_preferences[user_id] = prefs

# ---------- Memory ----------
def save_user_memory(user_id: str, memory: BaseChatMemory):
    os.makedirs(MEMORY_DIR, exist_ok=True)
    memory_vars = memory.load_memory_variables({})
    with open(os.path.join(MEMORY_DIR, f"{user_id}.pkl"), "wb") as f:
        pickle.dump(memory_vars, f)
    print(f"[MEMORY] Saved memory for {user_id}")

def load_user_memory(user_id: str) -> Optional[BaseChatMemory]:
    path = os.path.join(MEMORY_DIR, f"{user_id}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            memory_vars = pickle.load(f)
        print(f"[MEMORY] Loaded memory for {user_id}")
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            return_messages=True
        )
        if "history" in memory_vars:
            memory.chat_memory.messages = memory.chat_memory.messages + memory_vars["history"]
        return memory
    return None

def get_user_memory(user_id: str) -> BaseChatMemory:
    if user_id not in user_memories:
        memory = load_user_memory(user_id)
        if memory is None:
            memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=1000,
                return_messages=True
            )
        user_memories[user_id] = memory
    return user_memories[user_id]
# ---------- FAISS ----------
def save_faiss_to_disk(faiss_store: FAISS):
    with open(FAISS_PATH, "wb") as f:
        pickle.dump(faiss_store, f)
    print("[INFO] FAISS store saved to disk.")

def load_faiss_from_disk() -> Optional[FAISS]:
    if os.path.exists(FAISS_PATH):
        with open(FAISS_PATH, "rb") as f:
            faiss_store = pickle.load(f)
        print("[INFO] FAISS store loaded from disk.")
        return faiss_store
    print("[INFO] No FAISS store found on disk.")
    return None

# ---------- Node Functions ----------
def intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("input", "")
    user_id = state.get("user_id", "")
    print(f"[NODE] Intent detection - Query: {query}")

    memory = get_user_memory(user_id)
    memory_summary = memory.load_memory_variables({}).get("history", "")

    prompt = f"""You are an advanced assistant. Given the user's input and prior conversation, classify the intent and rewrite the input if necessary.

### Intent Categories:
1. search_papers — If the user is requesting academic papers.
2. general_question — If the user asks a general question unrelated to earlier context.
3. refine_question — If the user's query depends on prior discussion or context.

### Input:
History:
{memory_summary}

User Input:
{query}

### Format:
<INTENT>intent_name</INTENT>
<QUERY>rewritten standalone version of query if needed, otherwise just return original</QUERY>
"""

    llm_output = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    intent = "general_question"
    optimized_query = query

    if "<INTENT>" in llm_output and "</INTENT>" in llm_output:
        intent = llm_output.split("<INTENT>")[1].split("</INTENT>")[0].strip()
    if "<QUERY>" in llm_output and "</QUERY>" in llm_output:
        optimized_query = llm_output.split("<QUERY>")[1].split("</QUERY>")[0].strip()

    # ✨ Extract keywords and update contextual profile
    keywords = extract_keywords_from_query(optimized_query)
    update_contextual_profile(user_id, optimized_query, intent, keywords)

    return {
        **state,
        "intent": intent,
        "optimized_query": optimized_query,
        "original_query": query,
        "keywords": keywords,
    }

def paper_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("optimized_query", "")
    query = query.lower().split("and")[0]
    print(f"[NODE] Arxiv Search - Query: {query}")

    try:
        loader = ArxivLoader(query=query, load_max_docs=5)
        documents = loader.load()
        print(f"Number of documents loaded: {len(documents)}")
    except Exception as e:
        print(f"[ERROR] Arxiv load failed: {e}")
        documents = []

    if documents:
        papers_summary = "\n".join(
            f"- <https://arxiv.org/abs/{doc.metadata.get('arxiv_id') or doc.metadata.get('ArxivId') or extract_arxiv_id(doc.metadata.get('entry_id', ''))}|{doc.metadata.get('Title', 'Unknown Title')}>"
            for doc in documents
        )

        faiss_store = FAISS.from_documents(documents, embedding=embeddings_model)
        return {**state, "papers": papers_summary, "faiss_store": faiss_store}

    # If no papers, try a DuckDuckGo search
    print("[INFO] No papers found. Falling back to DuckDuckGo.")
    web_summary = search_duckduckgo(query)
    return {**state, "papers": "No papers found.", "faiss_store": None, "web_summary": web_summary}
   
def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("input", "")
    print(f"[NODE] LLM Node - Query: {query}")
    response = llm.invoke([HumanMessage(content=f"Answer the user's query concisely: {query}")]).content
    return {**state, "llm_output": response}

def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("input", "")
    faiss_store: Optional[VectorStore] = state.get("faiss_store")
    user_id = state.get("user_id", "")
    print(f"[NODE] RAG Node - Query: {query}")

    preferences = get_user_preferences(user_id)
    profile = get_contextual_profile(user_id)

    # Combine static preferences and dynamic profile
    context_prefs = f"""User Interests: {preferences.get('interests')}
Research Focus: {preferences.get('focus')}
Topics of Interest: {profile.get('topics')}
Preferred Question Types: {profile.get('question_types')}
Recent Queries: {profile.get('last_queries')}
"""

    if faiss_store:
        docs = faiss_store.similarity_search(query, k=3)
        summarized_docs = [
            llm.invoke([HumanMessage(content=f"Summarize this content concisely:\n{doc.page_content}")]).content
            for doc in docs
        ]
        context = "\n".join(summarized_docs)
        prompt = f"""You are a research assistant. Use the user's dynamic and static preferences to tailor your answer.

### User Context:
{context_prefs}

### Paper Summaries:
{context}

### Question:
{query}

### Answer Style:
Be helpful, context-aware, and align your answer with the user's topic interests and preferred query styles.
"""

        response = llm.invoke([HumanMessage(content=prompt)]).content
        return {**state, "rag_output": response}
    else:
        return {**state, "rag_output": "No papers found for context."}

def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("input", "")
    user_id = state.get("user_id", "")
    print(f"[NODE] Refine Node (Follow up) - Query: {query}")

    memory = get_user_memory(user_id)
    memory_summary = memory.load_memory_variables({}).get("history", "")
    prompt = f"Context from prior discussion:\n{memory_summary}\n\nFollow-up question:\n{query}"
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "refined_output": response}

def implementation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_id = state.get("user_id", "")
    user_query = state.get("input", "")
    rag_info = state.get("rag_output", "")
    llm_info = state.get("llm_output", "")
    web_summary = state.get("web_summary", "")
    fallback_context = rag_info or llm_info or web_summary or "No context available."
    preferences = get_user_preferences(user_id)

    prompt = f"""You are an AI research assistant. Based on the following context and user preferences, suggest practical implementations, experiments, or code ideas that the user can try.

### User Preferences:
Interests: {preferences.get('interests')}
Research Focus: {preferences.get('focus')}

### User Query:
{user_query}

### Context from Papers or Summary:
{fallback_context}

### Format:
Give 1-3 implementation ideas with clear steps. Include code snippets or pseudocode if relevant.
"""

    suggestion = llm.invoke(prompt)
    return {**state, "implementation_suggestions": suggestion}

# ---------- Answer Formatting ----------

def format_answer(response: str, papers_summary: str) -> str:
    """
    Format the final answer for the user, optionally including paper summaries.
    
    Args:
        response (str): The main answer generated by the model.
        papers_summary (str): A summary list of related papers, formatted with links.

    Returns:
        str: A well-formatted string with the answer and optional papers.
    """
    formatted = response.strip()
    
    if papers_summary and papers_summary != "No papers found.":
        formatted += "\n\n📚 **Related Papers:**\n" + papers_summary
    
    return formatted