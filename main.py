from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt import App
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Slack app
slack_app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

# Import and register event handlers
from slack_bot.events import register_listeners
register_listeners(slack_app)

# Initialize FastAPI app
app = FastAPI()

# Slack Request Handler
handler = SlackRequestHandler(slack_app)

# Slack endpoint for incoming events
@app.post("/slack/events")
async def endpoint(req: Request):
    return await handler.handle(req)
