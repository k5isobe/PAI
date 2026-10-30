
import json
import os
import sys
import urllib.request
import urllib.error

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY is not set", file=sys.stderr)
    sys.exit(1)

instruction = sys.argv[1] if len(sys.argv) > 1 else "右側に置く"

system_instruction = """
あなたはロボットアームの配置指示を解釈するAIです。
ユーザーの日本語指示から、箱の配置先座標 x, y を決定してください。

座標系と配置ルール:
- xの範囲: 0.10 ～ 0.15 m
- yの範囲: -0.05 ～ 0.05 m
- 右側: x=0.15, y=0.0
- 手前: x=0.10, y=-0.05
- 左側: x=0.10, y=0.05
- 奥側: x=0.15, y=0.05

複数の方向が指定された場合は、その意味を考慮して適切な座標を選んでください。
範囲外の座標は絶対に出さないでください。
JSONのみを返してください。
形式: {"x": 0.15, "y": 0.0}
"""

payload = {
    "system_instruction": {
        "parts": [{"text": system_instruction}]
    },
    "contents": [{
        "role": "user",
        "parts": [{"text": instruction}]
    }],
    "generationConfig": {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "x": {"type": "NUMBER"},
                "y": {"type": "NUMBER"}
            },
            "required": ["x", "y"]
        },
        "maxOutputTokens": 512,
        "thinkingConfig": {
            "thinkingLevel": "low"
        }
    }
}

url = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-3.8-flash:generateContent"
)

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    position = json.loads(text)

    x = float(position["x"])
    y = float(position["y"])

    # AIの出力をプログラム側でも検証
    if not (0.10 <= x <= 0.15 and -0.05 <= y <= 0.05):
        raise ValueError("AI returned coordinates outside allowed range")

    print(f"{x:.4f} {y:.4f}")

except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="replace")
    print(f"HTTP ERROR {e.code}: {detail}", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
