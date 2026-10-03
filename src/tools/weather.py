"""天气工具：查询指定地点当天天气（数据源 wttr.in，免费无需 API Key）。

网络错误时抛出异常（由上层 agent 或 eval 处理）。
"""

import json
import os
import urllib.parse
import urllib.request

# ── OpenAI function calling schema ──────────────────────────────────────────
TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": (
            "查询指定城市当天的实时天气，返回天气状况、当前气温、"
            "今日气温范围、湿度和风速。示例查询：'北京'、'Shanghai'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，中英文均可，如 '北京'、'Hangzhou'。",
                },
            },
            "required": ["city"],
        },
    },
}

# 请求超时（秒）
_TIMEOUT = 15
_USER_AGENT = "llm-beginner-task5/1.0 (student-project)"


def _get_proxy() -> str | None:
    """按优先级读取代理环境变量，一个都没有则返回 None（直连）。"""
    return (
        os.environ.get("WEATHER_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("WIKI_PROXY")
    )


def _fetch_weather(city: str) -> dict:
    """请求 wttr.in 的 JSON 接口，返回解析后的字典。网络失败抛异常。"""
    proxy = _get_proxy()
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    else:
        opener = urllib.request.build_opener()

    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1&lang=zh"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with opener.open(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(args: dict) -> str:
    city: str = args.get("city", "")
    if not city.strip():
        return '错误：缺少必填参数 "city"（城市名）。正确格式：{"city": "北京"}'

    try:
        data = _fetch_weather(city.strip())
    except Exception as exc:
        raise ConnectionError(f"天气查询失败：{exc}") from exc

    current = data["current_condition"][0]
    today = data["weather"][0]

    desc_list = current.get("lang_zh") or current.get("weatherDesc") or [{}]
    desc = desc_list[0].get("value", "未知")

    return "\n".join([
        f"城市：{city.strip()}",
        f"天气：{desc}",
        f"当前气温：{current.get('temp_C', '?')}°C（体感 {current.get('FeelsLikeC', '?')}°C）",
        f"今日气温：{today.get('mintempC', '?')}°C ~ {today.get('maxtempC', '?')}°C",
        f"湿度：{current.get('humidity', '?')}%，风速：{current.get('windspeedKmph', '?')} km/h",
    ])
