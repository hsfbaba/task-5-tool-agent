"""计算器工具：安全地计算数学表达式。

支持四则运算（+ - * / **）和高级函数（sqrt, sin, cos, log, pow, abs, pi, e 等）。
"""

import math

# ── OpenAI function calling schema ──────────────────────────────────────────
TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "计算数学表达式。支持四则运算（+ - * / **）和高级函数："
            "sqrt, sin, cos, tan, log, log2, log10, pow, abs, ceil, floor, "
            "exp, 以及常量 pi, e。示例表达式：'2 + 3 * 4'、'sqrt(16)'、'sin(pi/2)'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "要计算的数学表达式字符串，例如 '2 + 3 * 4'、"
                        "'(123 + 456) * 789'、'sqrt(2026)'。"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

# ── Safe evaluation helpers ─────────────────────────────────────────────────

# 允许的数学命名空间
_SAFE_NAMES: dict = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}

# 补充常用别名
_SAFE_NAMES["abs"] = abs
_SAFE_NAMES["round"] = round
_SAFE_NAMES["pow"] = pow
_SAFE_NAMES["int"] = int
_SAFE_NAMES["float"] = float


def run(args: dict) -> str:
    """安全计算数学表达式。

    Args:
        args: {"expression": "2 + 3 * 4"}

    Returns:
        计算结果字符串，或错误描述。
    """
    expression: str = args.get("expression", "")
    if not expression.strip():
        return "错误：expression 不能为空"

    try:
        # 在受限命名空间中 eval，禁止 builtins
        result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)
    except SyntaxError as e:
        return f"语法错误：{e}"
    except ZeroDivisionError:
        return "数学错误：除零"
    except Exception as e:
        return f"计算错误：{e}"

    # 统一格式化：整数值的 float 去掉小数点（is_integer 对 inf/nan 也安全）
    if isinstance(result, float):
        return str(int(result)) if result.is_integer() else f"{result:.10g}"
    return str(result)
