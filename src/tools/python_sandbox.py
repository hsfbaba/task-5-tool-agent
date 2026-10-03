"""Python 沙箱工具：在受限环境中执行 Python 代码。

安全措施：
- 白名单 import（math, json, re, itertools, collections, functools, random, statistics）
- 10 秒超时
- 捕获 stdout
- 禁止危险 builtins（open, exec, eval 嵌套, compile, __import__ 非白名单模块）
"""

import io
import sys
import threading

# ── OpenAI function calling schema ──────────────────────────────────────────
TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限 Python 沙箱中执行代码。允许的模块："
            "math, json, re, itertools, collections, functools, random, statistics。"
            "使用 print() 输出结果。超时 10 秒。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "要执行的 Python 代码。使用 print() 输出结果。"
                        "例如：'print(sum(range(10)))'。"
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

# ── 白名单 ──────────────────────────────────────────────────────────────────
_ALLOWED_IMPORTS: set[str] = {
    "math",
    "json",
    "re",
    "itertools",
    "collections",
    "functools",
    "random",
    "statistics",
}

_DANGEROUS_BUILTINS: set[str] = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "memoryview",
    "help",
}


def _make_restricted_builtins() -> dict:
    """构建受限的 __builtins__ 字典。"""
    import builtins as _builtins

    safe: dict = {}
    for name in dir(_builtins):
        if name.startswith("_"):
            continue
        if name in _DANGEROUS_BUILTINS:
            continue
        safe[name] = getattr(_builtins, name)

    # 使用受限的 __import__
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in _ALLOWED_IMPORTS:
            raise ImportError(f"不允许导入模块 '{name}'。允许的模块：{sorted(_ALLOWED_IMPORTS)}")
        return __import__(name, globals, locals, fromlist, level)

    safe["__import__"] = _restricted_import
    safe["__name__"] = "__main__"
    return safe


def run(args: dict) -> str:
    """在受限沙箱中执行 Python 代码。

    Args:
        args: {"code": "print(sum(range(10)))"}

    Returns:
        stdout 输出，或错误描述。
    """
    code: str = args.get("code", "")
    if not code.strip():
        return "错误：code 不能为空"
    if len(code) > 10000:
        return "错误：代码过长（>10000 字符）"

    timeout: int = args.get("timeout", 10)

    restricted_globals: dict = {
        "__builtins__": _make_restricted_builtins(),
    }

    captured = io.StringIO()
    result_container: dict = {"output": None, "error": None, "done": False}
    saved_stdout = sys.stdout

    def _target() -> None:
        try:
            sys.stdout = captured
            exec(code, restricted_globals, {})
            result_container["output"] = captured.getvalue()
        except Exception as exc:
            result_container["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            sys.stdout = saved_stdout
            result_container["done"] = True

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result_container["done"]:
        # Python 线程无法被强制杀死，超时线程仍在后台运行；
        # 必须在这里恢复 stdout，否则它会被永久劫持到 StringIO
        sys.stdout = saved_stdout
        return f"执行超时：代码运行超过 {timeout} 秒"

    if result_container["error"] is not None:
        return f"执行错误：{result_container['error']}"

    output: str | None = result_container["output"]
    if output and output.strip():
        return output.rstrip()
    return "代码执行成功，无输出。"
