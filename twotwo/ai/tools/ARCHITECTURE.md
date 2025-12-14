# TwoTwo Tool System Revamp

## Architecture

```
twotwo/ai/
├── tools/
│   ├── __init__.py         # Tool manager
│   ├── base.py             # Base tool class
│   └── search_tool.py      # Web search tool
├── llm.py
└── ...
```

## Components

### 1. Base Tool Class (`tools/base.py`)
- Abstract base class for all tools
- Defines: `name`, `description`, `tag`, `is_enabled()`, `execute(query)`
- Each tool self-describes for the system prompt

### 2. Tool Manager (`tools/__init__.py`)
- Discovers and manages all tools
- `get_enabled_tools()` - returns list of enabled tools
- `get_tools_prompt()` - generates system prompt section for active tools
- `extract_tool_call(response)` - detects tool usage in LLM response
- `execute_tool(name, query)` - runs the tool and returns results

### 3. Search Tool (`tools/search_tool.py`)
- Extends base tool
- Contains Brave Search logic (moved from search.py)
- Tag: `<search>query</search>`

### 4. System Prompt Format
```
You are TwoTwo, the helpful and concise A.I assistant...

Tools:
- **search**: Search the web for current information (weather, news, time, etc.)
  Usage: <search>your query</search>

When using tools, output the tool tag first, wait for results, then respond naturally.
```

## Changes Required
- [x] Create `tools/` folder structure
- [x] Create `base.py` with BaseTool class
- [x] Create `search_tool.py` (refactor from search.py)
- [x] Create tool manager in `__init__.py`
- [x] Update controller to use new tool system
- [x] Keep settings toggles working
