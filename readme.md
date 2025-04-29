# Your First AI Agent (Google ADK Tutorial): Step-by-Step Tutorial

In this tutorial, I’ll explain in simple terms what AI, AI agents, and workflows are, and then I’ll walk you through building your very first AI agent in Python using Google’s Agent Development Kit (ADK). By the end, you’ll understand the differences between these concepts and have a working content-assistant agent you can run from your terminal or a web interface.


## Key Concepts

In the world of modern automation and smart systems, it is helpful to consider three key building blocks: AI, AI agents, and workflows.

![Key Concepts](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-2.webp)



### AI (Artificial Intelligence)

Think of AI as the “brain” of technology. It learns patterns from data — text, images, or numbers — and uses them to translate sentences, recognize pictures, or predict the weather. On its own, AI only provides a narrow ability to think or understand; it doesn’t decide when or how to use that ability.

![AI (Artificial Intelligence)](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-3.webp)



**For instance:**

- **Spam filter** in your email: it “knows” patterns of junk mail and flags them for you.
- **Language translator** on your phone: type a sentence in English and it instantly translates it into French.
- **A photo app** that tags “beach,” “dog,” or “sunset” in your pictures automatically.

### Workflows

Workflows enable you to map out a flowchart of steps that connect various apps and services. You select a trigger, such as a new entry appearing in a spreadsheet, and then define a sequence of actions like sending an email, posting to a chat channel, or invoking an AI process to analyze the data. Once activated, the workflow follows those rules precisely — each step always leads to the next, with no decision-making beyond what you’ve laid out.

![Workflows](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-4.webp)



**For instance:**

- Whenever someone fills out your Google Sheet, automatically send a summary email to you (or your team).
- As soon as your favorite blog publishes a post, post the headline and link in your Slack channel.
- When a customer submits your web form, add their info to Salesforce and tag them “new lead.”Putting It All Together

### AI Agent

An AI agent is like giving that “brain” a body and a job. It can take in new information, such as your questions or the contents of a web page, use an AI model to decide what to do, and then run code, call APIs, send an email, or update a spreadsheet. In other words, an AI agent wraps an AI model in rules and tools so it can work for you automatically, planning multiple steps, adjusting if something goes wrong, and even choosing which tool to use next.

![AI Agent](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-5.webp)



**For instance:**

- **Smart inbox assistant**: it reads new emails, summarizes them, drafts replies, and even sends follow-ups if no one responds.
- **Travel planner bot**: you say “book me a weekend in Chiang Mai,” it searches flights, picks the best price, reserves a hotel, and emails you the itinerary.
- **Social-media manager**: it watches for brand mentions, analyzes sentiment, then posts thank-you replies or escalates complaints to the support team.

### Putting It All Together

- **AI alone** is the smart engine (like a chess computer that knows how to play).
- **An AI agent** is that engine plus instructions and tools (like a chess-playing robot that can decide which tournament to enter, book its own travel, and adjust its strategy if the Wi-Fi drops).
- **A workflow** is a fixed recipe you build by hand (like a conveyor belt in a factory: if one widget arrives, do steps 1→2→3 every time, no surprises).

Okay, now that we have more clarity on these, let’s dive into Google’s ADK.

## What Is Google’s Agent Development Kit?

[**Google’s Agent Development Kit (ADK)**](https://google.github.io/adk-docs/tutorials/) is an open-source Python toolkit that makes it easier to create AI agents. Think of ADK as a collection of libraries and tools that handle a lot of the heavy lifting so that we can focus on an agent’s logic and abilities instead of low-level details.

![Google’s Agent Development Kit](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-6.webp)



**Key features of ADK:**

- **Flexible & Modular:** Start simple or scale to multi-agent workflows without rewriting core code.
- **Code-First:** Define your agent’s logic, tools, and tests entirely in Python.
- **Tool Integration:** Plug in any Python function (API call, data fetch, computation) as an agent capability.
- **Model-Agnostic:** Use Google’s Gemini models or any LLM (OpenAI, Anthropic, Meta) via LiteLLM.
- **Orchestration & Memory:** Built-in context tracking, multi-turn conversation handling, and session memory.

In summary, ADK provides a robust foundation for building an AI agent quickly and reliably. Now, let’s prepare our environment and develop our first agent.

## Build a Content Assistant AI Agent

We are going to build an AI Agent that will help us brainstorm topic ideas, then draft the article content, and then create a formatted draft in markdown. This agent has multiple sub-agents that work together.

Here’s how it works:

- **Idea Agent** — comes up with topic ideas.
- **Writer Agent** — turns ideas into a draft.
- **Formatter Agent** — turns the draft into something ready to publish.

Each sub-agent does its part. The main agent runs them all in order.

### Setup and Prerequisites

- Download VS Code or another IDE.
- **Install Python and PIP** if you don’t have it.
- **Create and activate a virtual environment** in your project folder:

```bash
python3 -m venv venv  
source venv/bin/activate   # macOS/Linux  
.\venv\Scripts\activate    # Windows
```

- Install Google’s ADK and Dependencies

```bash
pip install google-generativeai google-ad-agents
```

- Create a project folder for your agent.

```bash
mkdir adk-example  
cd adk-example
```

- Get API Key for Gemini
- Go to [https://aistudio.google.com/](https://aistudio.google.com/), sign in, and click Get API Key.
- Create a new key and copy it — treat it like a password.
- In your project folder, create a file named .env:

```python
GEMINI_API_KEY=YOUR_API_KEY_HERE
```
- Create an __**init__.py** file and add the following content:

```python
# __init__.py  
from . import agent
```

- Create an **agent.py** file.

Your project structure should look like this:

```bash
my-agent-project/  
├── agent.py  
├── __init__.py  
└── .env
```

- Copy the following code inside your agent.py file:

```python
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
```


Ok, your agent is ready, and now you can run it inside the terminal.

In your project’s parent folder, start the agent with:

```bash
adk run [your-project-folder]
```

You’ll see each step’s output in your terminal.

![AI Agent in your terminal](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-7.webp)



If you want to use a web UI, you can run the following command from the parent folder:

```bash
adk web
```

Then open the URL shown in your browser and select your project (e.g., **adk-example**) in the top left menu.

![Web UI](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-8.webp)



Enter your prompt and watch each agent step appear in real time. :)

![Web UI](https://proflead.dev/assets/blog/ai-agent-explained-google-adk/ai-agent-explained-google-adk-9.webp)


All the code you can download from my GitHub repository — [how to build an AI agent](https://github.com/proflead/how-to-build-ai-agent).

The full docs of Google’s ADK can be found [here](https://google.github.io/adk-docs/).

## How to Build an AI Agent: A Video Tutorial

I also created a video tutorial where I take you step by step through each step.



_Watch on YouTube:_ [_AI Agent Explained: How to Build an AI Agent with Google ADK_](https://youtu.be/yVIWyKJPTKo?si=-XUdERnkjL0QcPFX)

## Conclusion

Now you know the difference between AI, workflows, and AI agents — and you’ve built a simple content-assistant agent using Google’s ADK and Python. If you try this out or extend it, I’d love to see what you come up with! Share your projects or questions in the comments.

Thanks for reading, and happy coding!

Cheers! ;)
