from slack_bolt import App
from slack_sdk.errors import SlackApiError
from slack_bot.langgraph_flow import research_flow
from slack_bot.utils import (
    save_user_memory,
    get_user_memory,
    save_faiss_to_disk,
    load_faiss_from_disk,
    get_user_preferences,
    update_user_preferences,
    format_answer,
)
import time
from collections import deque

# Cache last N event_ts values
recent_event_ts = deque(maxlen=100)

def is_duplicate(event):
    ts = event.get("event_ts")
    if ts in recent_event_ts:
        return True
    recent_event_ts.append(ts)
    return False

MAX_HISTORY_LENGTH = 50
MAX_SLACK_MESSAGE_LENGTH = 4000  # Slack's max message length

def register_listeners(app: App):
    @app.event("message")
    def handle_message(event, say):
        if event.get("subtype") == "bot_message":
            return
        
        if is_duplicate(event):
            return

        user_query = event.get("text", "")
        user_id = event.get("user")

        if not user_query or not user_id:
            return

        # Handle preference updates
        if user_query.startswith("!prefs"):
            try:
                new_prefs = eval(user_query.replace("!prefs", "").strip())
                update_user_preferences(user_id, new_prefs)
                say(f"✅ Preferences updated: {get_user_preferences(user_id)}")
            except Exception as e:
                say(f"⚠️ Failed to update preferences: {e}")
            return

        # Add user message to memory
        memory = get_user_memory(user_id)
        memory.chat_memory.add_user_message(user_query)

        initial_state = {
            "input": user_query,
            "user_id": user_id,
            "intent": None,
            "papers": None,
            "faiss_store": load_faiss_from_disk(),
            "llm_output": None,
            "rag_output": None,
            "refined_output": None,
        }

        try:
            placeholder = app.client.chat_postMessage(
                channel=event["channel"],
                text="💡 Thinking...",
                thread_ts=event.get("ts")  # keep in thread
            )

            result = research_flow.invoke(initial_state)

            final_answer = result.get("rag_output") or result.get("llm_output") or result.get("refined_output") or "Sorry, no answer available."
            papers = result.get("papers", "")
            reply_text = format_answer(final_answer, papers)

            # Truncate message if it exceeds Slack's limit
            print("Length of reply_text:", len(reply_text))
            if len(reply_text) > MAX_SLACK_MESSAGE_LENGTH:
                reply_text = reply_text[:MAX_SLACK_MESSAGE_LENGTH - 30] + "… (message truncated)"
            print(f"Length of reply_text after truncation:", len(reply_text))

            memory.chat_memory.add_ai_message(final_answer)
            save_user_memory(user_id, memory)

            if result.get("faiss_store"):
                save_faiss_to_disk(result["faiss_store"])

            app.client.chat_update(
                channel=event["channel"],
                ts=placeholder["ts"],
                text=reply_text
            )

        except SlackApiError as slack_err:
            print(f"[SLACK ERROR] {slack_err}")
        except Exception as e:
            print(f"[ERROR] Research flow failed: {e}")
            try:
                app.client.chat_update(
                    channel=event["channel"],
                    ts=placeholder["ts"],
                    text="⚠️ Sorry, something went wrong."
                )
            except Exception as update_error:
                print(f"[ERROR] Failed to update Slack message: {update_error}")
