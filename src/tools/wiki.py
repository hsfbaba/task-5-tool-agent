"""维基百科查询工具：通过 wikipediaapi 获取页面摘要。

网络不可用时抛出异常（由上层 agent 或 eval 处理）。
"""

import os
import threading

# ── OpenAI function calling schema ──────────────────────────────────────────
TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科页面摘要。支持中英文等多语言。"
            "返回页面标题和摘要（最多 1500 字符）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "要查询的关键词或页面标题，例如 'Alan Turing'、"
                        "'图灵机'、'Geoffrey Hinton'。"
                    ),
                },
                "lang": {
                    "type": "string",
                    "description": "语言代码：'zh' 中文，'en' 英文。默认 'zh'。",
                    "enum": ["zh", "en"],
                },
            },
            "required": ["query"],
        },
    },
}

# 摘要最大字符数
_MAX_SUMMARY_CHARS = 1500
# 默认请求超时（秒）
_DEFAULT_TIMEOUT = 15


def _query_wiki(query: str, lang: str) -> tuple[str | None, str | None, str | None]:
    """实际执行维基百科查询（在子线程中运行）。

    Returns:
        (title, summary, error_msg)。成功时 error_msg 为 None；失败时 title/summary 为 None。
    """
    try:
        import wikipediaapi
    except ImportError as e:
        raise ImportError(
            "未安装 wikipedia-api，请执行 pip install wikipedia-api"
        ) from e

    user_agent = "llm-beginner-task5/1.0 (student-project; contact@example.com)"
    # wikipediaapi 内部自建不带代理的 httpx.Transport，环境变量代理对它无效，
    # 必须显式传入 proxy 参数（优先级：WIKI_PROXY > HTTPS_PROXY > HTTP_PROXY）
    proxy = (
        os.environ.get("WIKI_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
    )
    client_kwargs: dict = {"proxy": proxy} if proxy else {}
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent, **client_kwargs)

    try:
        page = wiki.page(query)
    except Exception as e:
        raise ConnectionError(f"维基百科请求失败：{e}") from e

    if not page.exists():
        # 尝试英文 fallback
        if lang != "en":
            try:
                wiki_en = wikipediaapi.Wikipedia(
                    language="en", user_agent=user_agent, **client_kwargs
                )
                page_en = wiki_en.page(query)
                if page_en.exists():
                    return page_en.title, page_en.summary, None
            except Exception:
                pass
        return None, None, (
            f"未找到页面 '{query}'（语言：{lang}）。"
            f"请尝试更精确的关键词或换用英文搜索（指定 lang='en'）。"
        )

    return page.title, page.summary, None


def run(args: dict) -> str:
    """查询维基百科页面（带超时）。

    Args:
        args: {"query": "Alan Turing", "lang": "zh"}  lang 可选，默认 'zh'。

    Returns:
        页面标题 + 摘要。

    Raises:
        ImportError: wikipedia-api 未安装。
        TimeoutError: 请求超时。
        ConnectionError: 网络不可用。
    """
    query: str = args.get("query", "")
    if not query.strip():
        return '错误：缺少必填参数 "query"（要查询的关键词）。正确格式：{"query": "图灵机", "lang": "zh"}'

    lang: str = args.get("lang", "zh")
    timeout: int = args.get("timeout", _DEFAULT_TIMEOUT)

    # 使用线程实现超时
    result: list = [None]
    error: list[Exception | None] = [None]

    def _target() -> None:
        try:
            result[0] = _query_wiki(query, lang)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"维基百科查询超时（{timeout} 秒）")

    if error[0] is not None:
        raise error[0]  # 向上抛出 ImportError / ConnectionError

    title, summary, err_msg = result[0]

    if err_msg is not None:
        return err_msg

    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "..."

    return f"标题：{title}\n\n摘要：{summary}"
