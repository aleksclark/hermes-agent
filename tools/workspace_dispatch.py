"""Workspace Dispatch — create awesometree workspaces and orchestrate Crush agents.

Bridges awesometree workspace management with ACP agent communication.
Supports parallel dispatch to multiple workspaces with fire-and-forget
or wait-for-completion semantics.

Two tools:
  - workspace_dispatch: Create workspaces and send tasks to Crush agents
  - workspace_poll: Check on dispatched agent progress

Requires the awesometree daemon running on localhost:9099.
"""

import json
import logging
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)

_DAEMON_URL = "http://localhost:9099"
_DEFAULT_TIMEOUT = 10  # seconds for management API calls
_AGENT_STARTUP_WAIT = 6  # seconds to wait for Crush to start
_POLL_TIMEOUT = 5
_MAX_PARALLEL = 5

# In-flight dispatches keyed by workspace name for later polling
_dispatches: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api(method: str, path: str, json_body: dict = None, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Make an HTTP request to the awesometree daemon."""
    url = f"{_DAEMON_URL}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                resp = client.get(url)
            elif method == "POST":
                resp = client.post(url, json=json_body or {})
            elif method == "DELETE":
                resp = client.delete(url)
            elif method == "PUT":
                resp = client.put(url, json=json_body or {})
            else:
                return {"error": f"Unknown method: {method}"}

            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                return {"error": f"HTTP {resp.status_code}: {body}"}
            try:
                return resp.json() if resp.text else {}
            except Exception:
                return {"text": resp.text}
    except httpx.ConnectError:
        return {"error": "Cannot connect to awesometree daemon at localhost:9099. Is it running?"}
    except Exception as exc:
        return {"error": str(exc)}


def _ws_encode(name: str) -> str:
    """URL-encode a workspace name for API paths."""
    return urllib.parse.quote(name, safe="")


def _wait_for_health(workspace: str, timeout: int = _AGENT_STARTUP_WAIT) -> bool:
    """Poll the ACP health endpoint until the agent is ready."""
    ws = _ws_encode(workspace)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _api("GET", f"/api/acp/{ws}/health")
        if result.get("status") == "ok":
            return True
        time.sleep(1)
    return False


def _send_task(workspace: str, message: str, timeout: int = 300) -> dict:
    """Send a task via the /send endpoint (blocking)."""
    ws = _ws_encode(workspace)
    return _api("POST", f"/api/acp/{ws}/send",
                json_body={"message": message}, timeout=timeout)


def _send_task_fire_and_forget(workspace: str, message: str) -> dict:
    """Send a task via /stream, immediately disconnect. Agent continues working."""
    ws = _ws_encode(workspace)
    url = f"{_DAEMON_URL}/api/acp/{ws}/stream"
    try:
        with httpx.Client(timeout=httpx.Timeout(15, connect=5)) as client:
            # Open the stream, read just enough to confirm delivery, then close
            with client.stream("POST", url,
                               json={"message": message},
                               headers={"Accept": "text/event-stream",
                                        "Content-Type": "application/json"}) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode(errors="replace")
                    return {"error": f"HTTP {resp.status_code}: {body}"}
                # Read first few bytes to confirm the stream started
                for _ in resp.iter_lines():
                    break  # one line is enough
                return {"status": "dispatched"}
    except httpx.ReadTimeout:
        # Read timeout on stream is fine — it means the stream started
        return {"status": "dispatched"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to awesometree daemon"}
    except Exception as exc:
        return {"error": str(exc)}


def _get_history(workspace: str) -> dict:
    """Get the conversation history for a workspace's agent."""
    ws = _ws_encode(workspace)
    result = _api("GET", f"/api/acp/{ws}/history")
    if isinstance(result, list):
        return {"messages": result}
    return result


# ---------------------------------------------------------------------------
# Core dispatch logic
# ---------------------------------------------------------------------------

def _dispatch_one(task: dict) -> dict:
    """Dispatch a single task to a workspace. Creates the workspace if needed.

    Returns a result dict with status and details.
    """
    workspace = task.get("workspace", "")
    project = task.get("project", "")
    message = task.get("message", "")
    wait = task.get("wait", False)
    wait_timeout = task.get("wait_timeout", 300)

    if not workspace:
        return {"error": "workspace name is required"}
    if not message:
        return {"error": "message is required"}

    result = {"workspace": workspace}

    # Step 1: Check if workspace exists, create if project is specified
    ws_info = _api("GET", f"/api/workspaces/{_ws_encode(workspace)}")
    if "error" in ws_info:
        if not project:
            return {"workspace": workspace,
                    "error": f"Workspace '{workspace}' not found. Specify 'project' to create it."}
        # Create it
        create_result = _api("POST", "/api/workspaces",
                             json_body={"name": workspace, "project": project})
        if "error" in create_result:
            return {"workspace": workspace, "error": create_result["error"]}
        ws_info = create_result
        result["created"] = True
        logger.info("Created workspace %s for project %s", workspace, project)

    result["dir"] = ws_info.get("dir", "")
    result["acp_port"] = ws_info.get("acp_port")
    result["acp_status"] = ws_info.get("acp_status")

    # Step 2: Ensure the agent is running
    if not _wait_for_health(workspace, timeout=_AGENT_STARTUP_WAIT):
        # Try manual start if the daemon didn't auto-start Crush
        port = ws_info.get("acp_port")
        worktree_dir = ws_info.get("dir", "")
        if port and worktree_dir:
            logger.info("Agent not healthy, attempting manual Crush start on port %s", port)
            os.system(
                f"CRUSH_ACP_PORT={port} nohup crush serve --cwd {worktree_dir} "
                f"> /tmp/crush-{_ws_encode(workspace)}.log 2>&1 &"
            )
            if not _wait_for_health(workspace, timeout=10):
                return {**result, "error": "Agent failed to start. Check /tmp/crush-*.log"}
        else:
            return {**result, "error": "Agent not healthy and cannot determine port/dir for manual start"}

    result["acp_status"] = "running"

    # Step 3: Send the task
    if wait:
        send_result = _send_task(workspace, message, timeout=wait_timeout)
        if "error" in send_result:
            result["error"] = send_result["error"]
            result["status"] = "failed"
        else:
            result["status"] = "completed"
            result["session_id"] = send_result.get("session_id", "")
            result["response"] = send_result.get("text", "")
    else:
        send_result = _send_task_fire_and_forget(workspace, message)
        if "error" in send_result:
            result["error"] = send_result["error"]
            result["status"] = "failed"
        else:
            result["status"] = "dispatched"

    # Track for later polling
    _dispatches[workspace] = {
        "dispatched_at": time.time(),
        "message": message[:200],
        "status": result.get("status", "unknown"),
    }

    return result


# ---------------------------------------------------------------------------
# Tool: workspace_dispatch
# ---------------------------------------------------------------------------

def workspace_dispatch(tasks: list, task_id: str = None) -> str:
    """Dispatch tasks to one or more awesometree workspaces."""
    if not tasks:
        return json.dumps({"error": "tasks array is required and must not be empty"})

    if len(tasks) > _MAX_PARALLEL:
        return json.dumps({"error": f"Maximum {_MAX_PARALLEL} parallel tasks"})

    if len(tasks) == 1:
        # Single task — run inline
        result = _dispatch_one(tasks[0])
        return json.dumps(result, indent=2, ensure_ascii=False)

    # Parallel dispatch
    results = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), _MAX_PARALLEL)) as pool:
        futures = {pool.submit(_dispatch_one, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                task = futures[future]
                results.append({
                    "workspace": task.get("workspace", "?"),
                    "error": str(exc),
                    "status": "failed",
                })

    return json.dumps(results, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: workspace_poll
# ---------------------------------------------------------------------------

def workspace_poll(
    workspaces: list,
    wait: bool = False,
    wait_timeout: int = 120,
    task_id: str = None,
) -> str:
    """Poll one or more workspaces for agent progress."""
    if not workspaces:
        # Return all tracked dispatches
        if not _dispatches:
            return json.dumps({"message": "No active dispatches being tracked."})
        summaries = []
        for ws, info in _dispatches.items():
            summaries.append({"workspace": ws, **info})
        return json.dumps(summaries, indent=2, ensure_ascii=False)

    results = []
    for ws_name in workspaces:
        ws_result = {"workspace": ws_name}

        # Health check
        health = _api("GET", f"/api/acp/{_ws_encode(ws_name)}/health")
        ws_result["healthy"] = health.get("status") == "ok"

        # Get history
        history = _get_history(ws_name)
        if "error" in history:
            ws_result["error"] = history["error"]
            results.append(ws_result)
            continue

        messages = history.get("messages", history if isinstance(history, list) else [])
        agent_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "agent"]

        ws_result["total_messages"] = len(messages)
        ws_result["agent_messages"] = len(agent_msgs)

        # Include the last few agent messages as a progress summary
        recent = agent_msgs[-3:] if agent_msgs else []
        ws_result["recent_agent_output"] = []
        for msg in recent:
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            ws_result["recent_agent_output"].append(content)

        # Check if the agent appears to be done (heuristic: last message mentions
        # commit, push, "done", "complete", or has no tool calls pending)
        if agent_msgs:
            last = agent_msgs[-1].get("content", "").lower()
            done_signals = ["committed", "pushed", "complete", "done", "finished",
                            "all tests pass", "all .* pass"]
            ws_result["appears_done"] = any(sig in last for sig in done_signals)
        else:
            ws_result["appears_done"] = False

        results.append(ws_result)

    if wait and not all(r.get("appears_done") for r in results):
        # Poll until all appear done or timeout
        deadline = time.monotonic() + wait_timeout
        poll_interval = 15
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 60)  # back off

            all_done = True
            for i, ws_name in enumerate(workspaces):
                if results[i].get("appears_done"):
                    continue
                history = _get_history(ws_name)
                messages = history.get("messages", history if isinstance(history, list) else [])
                agent_msgs = [m for m in messages
                              if isinstance(m, dict) and m.get("role") == "agent"]
                results[i]["total_messages"] = len(messages)
                results[i]["agent_messages"] = len(agent_msgs)

                recent = agent_msgs[-3:] if agent_msgs else []
                results[i]["recent_agent_output"] = []
                for msg in recent:
                    content = msg.get("content", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    results[i]["recent_agent_output"].append(content)

                if agent_msgs:
                    last = agent_msgs[-1].get("content", "").lower()
                    done_signals = ["committed", "pushed", "complete", "done",
                                    "finished", "all tests pass"]
                    results[i]["appears_done"] = any(sig in last for sig in done_signals)

                if not results[i].get("appears_done"):
                    all_done = False

            if all_done:
                break

    return json.dumps(results, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Check requirements
# ---------------------------------------------------------------------------

def _check_requirements() -> bool:
    """Return True if the awesometree daemon is reachable."""
    try:
        resp = httpx.get(f"{_DAEMON_URL}/api/projects", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Register tools
# ---------------------------------------------------------------------------

_TOOLSET = "acp"

registry.register(
    name="workspace_dispatch",
    toolset=_TOOLSET,
    schema={
        "name": "workspace_dispatch",
        "description": (
            "Dispatch coding tasks to Crush AI agents in awesometree workspaces. "
            "Creates workspaces if they don't exist, waits for the agent to start, "
            "and sends the task. Supports parallel dispatch to multiple workspaces. "
            "\n\nTwo modes per task:"
            "\n- wait=false (default): Fire-and-forget. Returns immediately after dispatching. "
            "Use workspace_poll later to check progress."
            "\n- wait=true: Blocks until the agent completes and returns the response."
            "\n\nEach task in the array specifies: workspace name, project (for creation), "
            "message (the task), and whether to wait."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Array of task objects to dispatch. Up to 5 tasks run in parallel.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "workspace": {
                                "type": "string",
                                "description": "Workspace name. If it doesn't exist and 'project' is given, it will be created.",
                            },
                            "project": {
                                "type": "string",
                                "description": "Project name (e.g. 'curri', 'hermes'). Required only when creating a new workspace.",
                            },
                            "message": {
                                "type": "string",
                                "description": "The task/prompt to send to the Crush agent.",
                            },
                            "wait": {
                                "type": "boolean",
                                "description": "If true, block until the agent responds. If false (default), fire-and-forget.",
                                "default": False,
                            },
                            "wait_timeout": {
                                "type": "integer",
                                "description": "Max seconds to wait when wait=true. Default 300.",
                                "default": 300,
                            },
                        },
                        "required": ["workspace", "message"],
                    },
                },
            },
            "required": ["tasks"],
        },
    },
    handler=lambda args, **kw: workspace_dispatch(
        tasks=args.get("tasks", []),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_requirements,
)

registry.register(
    name="workspace_poll",
    toolset=_TOOLSET,
    schema={
        "name": "workspace_poll",
        "description": (
            "Check on Crush agent progress in awesometree workspaces. "
            "Returns health status, message counts, recent agent output, and whether "
            "the agent appears to be done."
            "\n\nTwo modes:"
            "\n- Poll (default): Returns current status immediately."
            "\n- Wait: Polls repeatedly until all agents appear done or timeout is reached."
            "\n\nCall with no workspaces to see all tracked dispatches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workspaces": {
                    "type": "array",
                    "description": "Workspace names to check. Omit to list all tracked dispatches.",
                    "items": {"type": "string"},
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll repeatedly until all agents appear done. Default false.",
                    "default": False,
                },
                "wait_timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait when wait=true. Default 120.",
                    "default": 120,
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: workspace_poll(
        workspaces=args.get("workspaces", []),
        wait=args.get("wait", False),
        wait_timeout=args.get("wait_timeout", 120),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_requirements,
)
