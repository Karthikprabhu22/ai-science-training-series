from typing import TypedDict, Annotated

from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from inference_auth_token import get_access_token

from yts_tools import download_youtube_video

# ============================================================
# 1. STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. ROUTE TOOLS
# ============================================================

def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    msgs = state["messages"]
    ai_msg = msgs[-1]

    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return "tools"
    return "done"


# ============================================================
# 3. AGENT 1 YT VIDEO DOWNLOADER
# ============================================================

def yts_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = (
        "You are an assistant that downloads YouTube videos using tools."
        "If user gives a link, call the downloader tool."
    ),
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    llm_with_tools = llm.bind_tools(tools=tools)
    out = llm_with_tools.invoke(messages)
    return {"messages": [out]}


# ============================================================
# 4. AGENT 2 CLIP RECOMMENDER
# ============================================================
def clip_recommendation_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = (
        "You are a YouTube Shorts clipping assistant.\n\n"
        "INPUT:\n"
        "- You will receive a single JSON object with keys such as "
        "'title', 'description', 'duration', 'heatmap', and 'video_path'.\n"
        "- 'heatmap' is a list of objects with fields 'start_time', 'end_time', and 'value', "
        "where larger 'value' means higher engagement.\n\n"
        "YOUR GOAL:\n"
        "- Choose ONE continuous time window within the video between 20 and 59 seconds long.\n"
        "- The window should cover a high-engagement region. Prefer one of the top 2-3 highest "
        "'value' heatmap bins and optionally include its neighbors to give musical context.\n"
        "- The clip should feel like a musically satisfying phrase (build-up + payoff), not just a random spike.\n\n"
        "ALGORITHM (follow this internally, but DO NOT describe it in your answer):\n"
        "1. Parse the 'heatmap' list from the input JSON.\n"
        "2. Sort heatmap entries by 'value' in descending order.\n"
        "3. Starting from the highest 'value', construct a candidate window of length 30-45 seconds\n"
        "   that includes that bin and, if possible, one or two neighboring bins.\n"
        "4. Ensure the window stays within [0, duration].\n"
        "5. If the best window is shorter than 20 seconds or longer than 59 seconds, adjust its start/end\n"
        "   while still covering the high-engagement region.\n\n"
        "OUTPUT FORMAT (MANDATORY):\n"
        "- Respond with a SINGLE valid JSON object and NOTHING else.\n"
        "- No prose, no explanation, no markdown, no code fences.\n"
        "- The JSON must have EXACTLY these keys:\n"
        "  {\n"
        "    \"start_time\": \"MM:SS\",\n"
        "    \"end_time\": \"MM:SS\",\n"
        "    \"suggested_title\": \"...\",\n"
        "    \"suggested_description\": \"...\",\n"
        "    \"suggested_hashtags\": [\"#tag1\", \"#tag2\", \"#tag3\"]\n"
        "  }\n"
        "- Use the original title/description/artist info as inspiration but keep it concise and optimized for Shorts.\n"
        "- 'start_time' and 'end_time' MUST be within the video 'duration'.\n"
        "- Do NOT include any other keys.\n"
    ),
):
    # 1. Find the latest tool message (output of download_youtube_video)
    tool_content = None
    for msg in reversed(state["messages"]):
        # LangChain tool messages usually have type == "tool"
        if getattr(msg, "type", None) == "tool":
            tool_content = msg.content
            break

    # Fallback: if not found, just send everything (worse, but still functional)
    if tool_content is None:
        tool_content = str(state["messages"])

    # 2. Build the messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Here is the video metadata as JSON. "
                "Use it to choose the best 20 - 59 second YouTube Short clip and "
                "return ONLY the JSON object as specified.\n\n"
                f"{tool_content}"
            ),
        },
    ]

    out = llm.invoke(messages)
    return {"messages": [out]}

# ============================================================
# 6. SETUP LLM AND TOOLS
# ============================================================
access_token = get_access_token()

llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

tools = [download_youtube_video]


# ============================================================
# 6. BUILD GRAPH
# ============================================================

graph_builder = StateGraph(State)

# Agent node: calls LLM, which may decide to call tools
graph_builder.add_node(
    "yts_agent",
    lambda s: yts_agent(s, llm=llm, tools=tools),
)

graph_builder.add_node(
    "clip_recommendation_agent",
    lambda s: clip_recommendation_agent(s, llm=llm),
)

# Tool node: executes tool calls emitted by the LLM
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Graph logic
# START -> yts_agent
graph_builder.add_edge(START, "yts_agent")

# After yts_agent runs, check if we need to run tools
graph_builder.add_conditional_edges(
    "yts_agent", route_tools,
    {"tools": "tools", "done": "clip_recommendation_agent"}
)

# After tools run, go back to the agent so it can use tool results
graph_builder.add_edge("tools", "clip_recommendation_agent")

# After structured_output_agent, terminate the graph
graph_builder.add_edge("clip_recommendation_agent", END)

graph = graph_builder.compile()


# ============================================================
# 7. RUN PIPELINE
# ============================================================

if __name__ == "__main__":
    prompt = "Download this video and suggest a short clip: https://www.youtube.com/watch?v=Uks8psEpmB4"

    for chunk in graph.stream({"messages": prompt}, stream_mode="values"):
        msg = chunk["messages"][-1]
        msg.pretty_print()

