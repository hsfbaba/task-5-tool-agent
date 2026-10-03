"""手写 ReAct Agent：Thought / Action / Action Input / Observation 循环。

通过 OpenAI 兼容 API 调用本地 LLM（Ollama / llama.cpp / vLLM），
让模型自主选择工具完成多步任务。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# ── 工具注册表 ──────────────────────────────────────────────────────────────
from src.tools.calculator import TOOL_SCHEMA as CALC_SCHEMA
from src.tools.calculator import run as calc_run
from src.tools.file_search import TOOL_SCHEMA as FILE_SCHEMA
from src.tools.file_search import run as file_run
from src.tools.python_sandbox import TOOL_SCHEMA as SANDBOX_SCHEMA
from src.tools.python_sandbox import run as sandbox_run
from src.tools.wiki import TOOL_SCHEMA as WIKI_SCHEMA
from src.tools.wiki import run as wiki_run

_TOOLS: dict[str, dict[str, Any]] = {
    "calculator":     {"schema": CALC_SCHEMA,     "run": calc_run},
    "python_sandbox": {"schema": SANDBOX_SCHEMA,  "run": sandbox_run},
    "file_search":    {"schema": FILE_SCHEMA,     "run": file_run},
    "wiki":           {"schema": WIKI_SCHEMA,     "run": wiki_run},
}

# ═══════════════════════════════════════════════════════════════════════════
# System prompt 模板
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """你是一个能够使用工具解决复杂任务的智能助手。你可以使用以下工具：

{tool_descriptions}

## 回复格式

你需要严格按照以下格式回复：

```
Thought: 你的推理过程——分析当前信息，决定下一步做什么
Action: 工具名称（必须是 [{tool_names}] 之一）
Action Input: 工具的 JSON 输入参数
```

工具执行后，你会收到：

```
Observation: 工具返回结果
```

你可以重复 Thought → Action → Action Input → Observation 多次。

当你已经收集到足够信息回答用户问题时，使用：

```
Thought: 我已经有足够的信息来回答
Final Answer: 你的最终答案
```

## 规则

1. 每次只能调用一个工具。
2. Action Input 必须是合法的 JSON 对象，如 {{"expression": "2+3"}}。
3. 如果工具返回错误，请分析原因并尝试修复或换用其他工具。
4. 不要猜测答案；使用工具获取或计算信息。
5. 如果工具多次尝试后仍失败（如网络超时），必须在 Final Answer 中如实说明工具调用失败及原因，禁止编造内容冒充查询结果。

## 示例

用户: 计算 (300 + 200) * 5 的结果
Thought: 我需要使用计算器来计算 (300 + 200) * 5。
Action: calculator
Action Input: {{"expression": "(300 + 200) * 5"}}
Observation: 2500
Thought: 计算结果已经得到，是 2500。
Final Answer: (300 + 200) * 5 = 2500

现在请解决以下任务。"""


def _build_system_prompt() -> str:
    """从工具注册表动态生成 system prompt。"""
    tool_descs: list[str] = []
    for name, info in _TOOLS.items():
        schema = info["schema"]
        func = schema["function"]
        desc = func.get("description", "")
        params = json.dumps(func.get("parameters", {}), ensure_ascii=False)
        tool_descs.append(f"- **{name}**: {desc}\n  参数 Schema：{params}")
    return _SYSTEM_PROMPT.format(
        tool_descriptions="\n".join(tool_descs),
        tool_names=", ".join(_TOOLS.keys()),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 响应解析
# ═══════════════════════════════════════════════════════════════════════════

# 匹配 Thought / Action / Action Input / Final Answer 的各个部分
_RE_THOUGHT = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final|Observation)|$)", re.DOTALL)
_RE_ACTION = re.compile(r"Action:\s*(\S+)")
_RE_ACTION_INPUT_MARK = re.compile(r"Action Input:\s*")
_RE_FINAL_ANSWER = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def _parse_response(text: str) -> dict[str, Any]:
    """解析模型输出的 ReAct 格式文本。

    Returns:
        {
            "type": "action" | "final_answer" | "parse_error",
            "thought": str,
            "action": str | None,      # 仅 action 类型
            "action_input": dict | None,  # 仅 action 类型
            "final_answer": str | None,   # 仅 final_answer 类型
            "raw": str,                   # 原始输出
        }
    """
    raw = text.strip()

    # 检测 Final Answer
    fa_match = _RE_FINAL_ANSWER.search(raw)
    if fa_match:
        thought_match = _RE_THOUGHT.search(raw)
        return {
            "type": "final_answer",
            "thought": thought_match.group(1).strip() if thought_match else "",
            "final_answer": fa_match.group(1).strip(),
            "raw": raw,
        }

    # 检测 Action
    action_match = _RE_ACTION.search(raw)
    if action_match:
        action = action_match.group(1).strip()
        if action not in _TOOLS:
            return {
                "type": "parse_error",
                "thought": "",
                "raw": raw,
                "error": f"未知工具 '{action}'。可用工具：{', '.join(_TOOLS.keys())}",
            }

        thought_match = _RE_THOUGHT.search(raw)
        thought = thought_match.group(1).strip() if thought_match else ""

        input_mark = _RE_ACTION_INPUT_MARK.search(raw)
        if not input_mark:
            return {
                "type": "parse_error",
                "thought": thought,
                "raw": raw,
                "error": "Action Input 格式错误：需要 JSON 对象，如 {\"key\": \"value\"}",
            }

        # 用 raw_decode 只解析第一个 JSON 对象，容忍模型在 JSON 后追加的多余文本
        try:
            action_input, _ = json.JSONDecoder().raw_decode(raw, input_mark.end())
        except json.JSONDecodeError as e:
            return {
                "type": "parse_error",
                "thought": thought,
                "raw": raw,
                "error": f"Action Input JSON 解析失败：{e}",
            }

        if not isinstance(action_input, dict):
            return {
                "type": "parse_error",
                "thought": thought,
                "raw": raw,
                "error": f"Action Input 必须是 JSON 对象，收到的是 {type(action_input).__name__}",
            }

        return {
            "type": "action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "raw": raw,
        }

    # 无法识别格式
    # 最后的 fallback：检查是否有类似 Final Answer 的自然语言
    if len(raw) > 10:
        return {
            "type": "parse_error",
            "thought": "",
            "raw": raw,
            "error": (
                "无法识别回复格式。请严格使用以下格式之一：\n"
                "1. Thought: ...\nAction: tool_name\nAction Input: {...}\n"
                "2. Thought: ...\nFinal Answer: ..."
            ),
        }

    return {
        "type": "parse_error",
        "thought": "",
        "raw": raw,
        "error": "回复为空或过短，请重新输出。",
    }


# ═══════════════════════════════════════════════════════════════════════════
# ReActAgent
# ═══════════════════════════════════════════════════════════════════════════

class ReActAgent:
    """手写 ReAct 循环 Agent。

    通过 OpenAI 兼容 API 调用 LLM，让模型在 Thought/Action/Observation 循环中
    自主选择工具完成任务。

    Usage:
        agent = ReActAgent(model="qwen2.5:7b-instruct")
        trace = agent.run("计算 2+3*4 的结果")
        print(trace["final_answer"])
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 10,
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            model: 模型名称。默认从 OPENAI_MODEL 环境变量读取，
                   回退到 "qwen2.5:7b-instruct"（Ollama 默认名）。
            base_url: API 地址。默认从 OPENAI_BASE_URL 环境变量读取，
                      回退到 "http://localhost:11434/v1"（Ollama 默认）。
            api_key: API key。默认从 OPENAI_API_KEY 环境变量读取，
                     回退到 "ollama"。
            max_steps: 最大 ReAct 步数，防止死循环。
            temperature: 采样温度，0.0 提高格式稳定性。
        """
        self.model = model or os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        self.max_steps = max_steps
        self.temperature = temperature
        self._system_prompt = _build_system_prompt()

        # 错误注入钩子（用于测试错误恢复能力）
        # 设置为 (tool_name, error_message) 时，下一次工具调用将被拦截
        self._inject_error: tuple[str, str] | None = None

        # 延迟导入 openai
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端，避免在仅做工具测试时导入失败。"""
        if self._client is None:
            from openai import OpenAI  # type: ignore[import-untyped]
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def inject_error(self, tool_name: str, error_message: str) -> None:
        """注入一个工具错误，用于测试 agent 的错误恢复能力。

        设置后，下一次工具调用将返回该错误而非实际执行。
        仅影响一次调用，之后自动清除。

        Args:
            tool_name: 目标工具名。
            error_message: 伪造的错误消息。
        """
        self._inject_error = (tool_name, error_message)

    # ── AgentTrace ──────────────────────────────────────────────────────

    def run(self, task: str) -> dict[str, Any]:
        """执行 ReAct 循环完成给定任务。

        Args:
            task: 用户任务描述。

        Returns:
            AgentTrace dict:
            {
                "steps": [
                    {
                        "thought": str,
                        "action": str,        # 工具名
                        "tool": str,          # 别名，同 action
                        "action_input": dict,
                        "observation": str,
                        "raw": str,           # 模型原始输出
                    },
                    ...
                ],
                "final_answer": str,
                "success": bool,
            }
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
        ]
        steps: list[dict[str, Any]] = []

        for step_idx in range(self.max_steps):
            # 调用 LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                raw_output = response.choices[0].message.content or ""
            except Exception as e:
                return {
                    "steps": steps,
                    "final_answer": f"LLM 调用失败：{e}",
                    "success": False,
                }

            # 解析响应（最多重试 2 次格式错误）
            parsed = _parse_response(raw_output)
            retries = 0
            while parsed["type"] == "parse_error" and retries < 2:
                # 将错误作为反馈发送给模型要求重试
                correction = (
                    f"你的上一次回复格式有误：{parsed.get('error', '未知错误')}\n\n"
                    "请严格按照以下格式重新输出：\n"
                    "Thought: <你的推理>\n"
                    "Action: <工具名>\n"
                    "Action Input: <JSON>\n\n"
                    "或：\n"
                    "Thought: <推理>\n"
                    "Final Answer: <答案>"
                )
                # 先把模型的错误输出补进历史，再发纠正指令，
                # 否则会出现两条连续 user 消息，破坏对话结构
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": correction})
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=1024,
                    )
                    raw_output = response.choices[0].message.content or ""
                except Exception as e:
                    return {
                        "steps": steps,
                        "final_answer": f"LLM 调用失败（重试中）：{e}",
                        "success": False,
                    }
                parsed = _parse_response(raw_output)
                retries += 1

            # --- Final Answer ---
            if parsed["type"] == "final_answer":
                answer = parsed["final_answer"]
                # 兜底：若本次任务所有工具调用都失败了，给答案打上"未经验证"标记
                tool_steps = [s for s in steps if s["action"] not in ("parse_error", "")]
                if tool_steps and all(
                    str(s["observation"]).startswith(("工具执行异常", "错误"))
                    for s in tool_steps
                ):
                    answer = (
                        "⚠️ 本次任务的所有工具调用均失败，以下答案未经工具验证，"
                        "可能来自模型自身记忆：\n" + answer
                    )
                return {
                    "steps": steps,
                    "final_answer": answer,
                    "success": True,
                }

            # --- Parse Error (所有重试用尽) ---
            if parsed["type"] == "parse_error":
                # 将 parse error 当作步骤记录下来
                steps.append({
                    "thought": parsed.get("thought", ""),
                    "action": "parse_error",
                    "tool": "parse_error",
                    "action_input": {},
                    "observation": parsed.get("error", "解析失败"),
                    "raw": parsed.get("raw", ""),
                })
                messages.append({
                    "role": "assistant",
                    "content": parsed.get("raw", ""),
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: 格式错误 —— {parsed.get('error', '')}",
                })
                continue

            # --- Action ---
            tool_name = parsed["action"]
            tool_input = parsed["action_input"]

            # 检查错误注入
            if (
                self._inject_error is not None
                and self._inject_error[0] == tool_name
            ):
                observation = self._inject_error[1]
                self._inject_error = None  # 仅影响一次
            else:
                # 执行工具
                try:
                    tool_fn = _TOOLS[tool_name]["run"]
                    observation = tool_fn(tool_input)
                except Exception as exc:
                    observation = f"工具执行异常：{type(exc).__name__}: {exc}"

            # 记录步骤
            step_record: dict[str, Any] = {
                "thought": parsed["thought"],
                "action": tool_name,
                "tool": tool_name,  # 别名，兼容 eval 的多字段检测
                "action_input": tool_input,
                "observation": str(observation),
                "raw": parsed.get("raw", ""),
            }
            steps.append(step_record)

            # 将本轮交互追加到消息历史
            messages.append({
                "role": "assistant",
                "content": parsed.get("raw", ""),
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        # 步数用尽
        return {
            "steps": steps,
            "final_answer": (
                f"达到最大步数限制 ({self.max_steps})，"
                f"共执行了 {len(steps)} 步操作。"
            ),
            "success": False,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI 快速测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "计算 2 + 3 * 4 的结果"
    print(f"任务: {task}")
    print("=" * 60)

    agent = ReActAgent()
    trace = agent.run(task)

    print(f"\n成功: {trace['success']}")
    print(f"步数: {len(trace['steps'])}")
    for i, step in enumerate(trace["steps"], 1):
        print(f"\n--- Step {i} ---")
        print(f"  Thought: {step.get('thought', '')[:100]}")
        print(f"  Action:  {step.get('action', '')}")
        print(f"  Input:   {json.dumps(step.get('action_input', {}), ensure_ascii=False)}")
        print(f"  Obs:     {str(step.get('observation', ''))[:200]}")
    print(f"\n最终答案: {trace['final_answer']}")
