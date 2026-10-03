"""文件搜索工具：按文件名 / 内容搜索本地文件，或读取文件内容。

安全措施：路径越界保护（所有路径解析后必须在指定目录内）。
"""

from pathlib import Path

# ── OpenAI function calling schema ──────────────────────────────────────────
TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "搜索本地文件系统：按文件名模式匹配、按文件内容搜索关键词、"
            "或读取指定文件的内容。所有操作限制在指定目录内。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dir": {
                    "type": "string",
                    "description": "搜索/操作的根目录（绝对路径或相对路径）。",
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "文件名匹配模式，如 '*.md' 查找所有 Markdown 文件、"
                        "'README.md' 精确匹配。支持 glob 通配符。"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "可选。按文件内容搜索的关键词，返回包含该词的所有文件路径。",
                },
                "read_file": {
                    "type": "string",
                    "description": (
                        "可选。要读取内容的文件路径（相对于 dir）。"
                        "返回文件内容（前 2000 字符）。"
                    ),
                },
            },
            "required": ["dir"],
        },
    },
}

# 最大返回文件数
_MAX_RESULTS = 30
# 读取文件的最大字符数
_MAX_READ_CHARS = 2000


def _safe_resolve(dir_path: str, rel_path: str | None = None) -> Path:
    """安全解析路径，确保不越界。

    Args:
        dir_path: 根目录。
        rel_path: 相对于根目录的可选子路径。

    Returns:
        解析后的绝对路径。

    Raises:
        ValueError: 路径越界或不存在。
    """
    base = Path(dir_path).resolve(strict=True)
    if rel_path is None:
        return base

    target = (base / rel_path).resolve()
    # 确保 target 在 base 之下
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"路径越界：'{rel_path}' 不在目录 '{dir_path}' 内")
    return target


def _search_by_pattern(base: Path, pattern: str) -> str:
    """按文件名模式搜索。"""
    matches: list[str] = []
    for file_path in base.rglob(pattern):
        if file_path.is_file():
            matches.append(str(file_path.relative_to(base)))
            if len(matches) >= _MAX_RESULTS:
                break

    if not matches:
        return f"未找到匹配模式 '{pattern}' 的文件（目录：{base}）"

    lines = [f"找到 {len(matches)} 个匹配 '{pattern}' 的文件："]
    lines.extend(f"  - {m}" for m in matches)
    return "\n".join(lines)


def _search_by_content(base: Path, keyword: str) -> str:
    """按文件内容关键词搜索。"""
    matches: list[str] = []
    keyword_lower = keyword.lower()

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if keyword_lower in text.lower():
            matches.append(str(file_path.relative_to(base)))
            if len(matches) >= _MAX_RESULTS:
                break

    if not matches:
        return f"未找到包含 '{keyword}' 的文件（目录：{base}）"

    lines = [f"找到 {len(matches)} 个包含 '{keyword}' 的文件："]
    lines.extend(f"  - {m}" for m in matches)
    return "\n".join(lines)


def _read_file_content(base: Path, rel_path: str) -> str:
    """读取文件内容。"""
    target = _safe_resolve(str(base), rel_path)
    if not target.exists():
        return f"文件不存在：{rel_path}"
    if not target.is_file():
        return f"路径不是文件：{rel_path}"

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return f"读取文件失败：{e}"

    if len(content) > _MAX_READ_CHARS:
        content = content[:_MAX_READ_CHARS] + f"\n\n... [截断，共 {len(content)} 字符]"
    return f"文件 {rel_path} 的内容：\n\n{content}"


# ── 主入口 ──────────────────────────────────────────────────────────────────

def run(args: dict) -> str:
    """搜索本地文件或读取文件内容。

    Args:
        args: {
            "dir": str (必填),
            "pattern": str (可选，文件名模式),
            "content": str (可选，内容关键词),
            "read_file": str (可选，要读取的文件路径),
        }

    Returns:
        搜索结果或文件内容字符串。
    """
    dir_path: str = args.get("dir", "")
    if not dir_path:
        return "错误：'dir' 参数不能为空"

    try:
        base = _safe_resolve(dir_path)
    except FileNotFoundError:
        return f"错误：目录不存在 —— {dir_path}"
    except ValueError as e:
        return f"错误：{e}"

    # 模式 1：读取文件
    read_file: str | None = args.get("read_file")
    if read_file:
        try:
            return _read_file_content(base, read_file)
        except ValueError as e:
            return f"错误：{e}"

    # 模式 2：按内容搜索
    content: str | None = args.get("content")
    if content:
        return _search_by_content(base, content)

    # 模式 3：按文件名模式搜索
    pattern: str = args.get("pattern", "*")
    return _search_by_pattern(base, pattern)
