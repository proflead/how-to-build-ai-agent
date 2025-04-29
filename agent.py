# agent.py

import os                     # Provides functions for interacting with the operating system\ nfrom dotenv import load_dotenv  # Loads environment variables from a .env file
from dotenv import load_dotenv
import google.generativeai as genai  # Official Google Generative AI SDK aliased as "genai"
from google.adk.agents import LlmAgent, SequentialAgent  # ADK classes for building language-model agents

# ——————————————————————————————————————————————
# 0) Load .env from this folder
# ——————————————————————————————————————————————
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))  # Read the .env file in the same directory as this script

# ——————————————————————————————————————————————
# 1) Configure the Google Gen AI SDK using your .env
# ——————————————————————————————————————————————
api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve the API key from environment variables
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")  # Halt if the key is not set
genai.configure(api_key=api_key)  # Initialize the generative-ai SDK with your key

# ——————————————————————————————————————————————
# 2) Tool functions that actually call Gemini
# ——————————————————————————————————————————————

def generate_ideas(topic: str) -> str:
    """Ask Gemini to brainstorm blog post idea for a topic."""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Create a Gemini model instance
    resp = model.generate_content(
        f"Brainstorm 4–6 creative blog post ideas for the topic:\n\n{topic}"
    )  # Generate the ideas
    return resp.text  # Return only the generated text


def write_content(ideas: str) -> str:
    """Ask Gemini to expand an outline into a ~300-word draft."""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Instantiate the model again
    resp = model.generate_content(
        "Expand the following outline into a cohesive ~300-word blog post:\n\n"
        f"{ideas}"
    )  # Generate the full draft
    return resp.text  # Return the draft text


def format_draft(draft: str) -> str:
    """Ask Gemini to format the draft as clean Markdown."""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # New model instance for formatting
    resp = model.generate_content(
        "Format this draft as clean Markdown with headings, sub-headings, and bullet lists:\n\n"
        f"{draft}"
    )  # Generate formatted Markdown
    return resp.text  # Return the formatted markdown

# ——————————————————————————————————————————————
# 3) Define your three LLM agents
# ——————————————————————————————————————————————

topic_agent = LlmAgent(
    name="IdeaAgent",  # Unique identifier for the agent
    model="gemini-2.0-flash",  # Model spec to use
    description="Brainstorms blog post ideas.",  # Purpose of this agent
    instruction=(
        "Call generate_ideas(topic) with the exact topic string you receive "
        "and return only the ideas."
    ),  # How the agent should behave
    tools=[generate_ideas],  # List of Python functions it can invoke
    output_key="ideas"  # Where to store the result in shared state
)

draft_agent = LlmAgent(
    name="WriterAgent",
    model="gemini-2.0-flash",
    description="Writes a blog post draft from ideas.",
    instruction=(
        "Call write_content(ideas), where `ideas` is the output from the prior step, "
        "and return only the draft text."
    ),
    tools=[write_content],
    output_key="draft"
)

format_agent = LlmAgent(
    name="FormatterAgent",
    model="gemini-2.0-flash",
    description="Formats the draft into Markdown.",
    instruction=(
        "Call format_draft(draft), where `draft` is the previous output, "
        "and return only the final Markdown."
    ),
    tools=[format_draft],
    output_key="formatted"
)

# ——————————————————————————————————————————————
# 4) Put them in a SequentialAgent pipeline
# ——————————————————————————————————————————————

root_agent = SequentialAgent(
    name="ContentAssistant",  # Top-level orchestration agent
    sub_agents=[topic_agent, draft_agent, format_agent],  # Steps in order
    description="Takes a user topic → generates ideas → writes draft → formats as Markdown"
)
