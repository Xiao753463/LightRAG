import json
from time import sleep
import ollama

# 定義 function schema
functions = [
    {
        "type": "function",
        "function": {
            "name": "analyze_review",
            "description": "分析評論中出現的所有對象、構面與情感，每個構面各回傳一筆資料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "評論主題，例如商品或服務"
                                },
                                "target": {
                                    "type": "string",
                                    "description": "評論中提及的對象"
                                },
                                "facet": {
                                    "type": "string",
                                    "description": "對該對象所評論的構面"
                                },
                                "sentiment": {
                                    "type": "number",
                                    "minimum": -1,
                                    "maximum": 1,
                                    "description": "評論情緒強度，負面為 -1，正面為 1，中立為 0，可包含中間數值"
                                }
                            },
                            "required": ["topic", "target", "facet", "sentiment"]
                        }
                    }
                },
                "required": ["results"]
            }
        }
    }
]


def build_prompt(review: str) -> str:
    return f"""
你是一位評論分析助手，負責將一段顧客評論拆解為多組結構化的面向，每一組都包含：

1. topic：主題（只能是「商品」或「服務」）
2. target：評論中提及的對象（例如：南瓜蛋糕、服務人員、產品）
   - ⚠️ 如果商品名稱中包含口味描述（如「蔓越莓口味酥餅」），請將「酥餅」視為 target，「蔓越莓口味」視為 facet（構面），不要把口味放進 target
3. facet：對該對象所描述的具體構面（例如：態度、多樣性、口感）
4. sentiment：對該構面的情緒傾向，請用數值表示，範圍從 -1 到 1

請從以下評論中，找出所有符合上述結構的資訊，每個構面輸出一筆資料：
評論：「{review}」

請直接以 JSON 格式回傳以下欄位：
results: [
  {{
    "topic": "...",
    "target": "...",
    "facet": "...",
    "sentiment": "..."
  }},
  ...
]
"""


def call_ollama(review: str):
    prompt = build_prompt(review)

    response = ollama.chat(
        model='llama3.1:8b-instruct-q5_K_M',  # 改成你支援 function calling 的模型
        messages=[{"role": "user", "content": prompt}],
        tools=functions
    )
    if "tool_calls" in response["message"]:
        for tool_call in response["message"]["tool_calls"]:
            args = tool_call.function.arguments
            raw_results = args["results"]
            if isinstance(raw_results, str):
                structured_results = json.loads(raw_results)
            else:
                structured_results = raw_results

            # 結構化輸出（可選印出）
            print("\n🎯 分析結果：")
            for i, item in enumerate(structured_results, 1):
                print(f"\n第 {i} 組結果：")
                print(f"  - 主題 (topic): {item['topic']}")
                print(f"  - 對象 (target): {item['target']}")
                print(f"  - 構面 (facet): {item['facet']}")
                print(f"  - 情感 (sentiment): {item['sentiment']}")
            return structured_results
    else:
        return {"error": "Function call failed or not triggered"}


def analyze_reviews_from_file(input_path: str, output_path: str):
    # 1. 讀入評論 JSON 檔案
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 對每筆評論進行分析
    for i, item in enumerate(data):
        review = item.get("review", "").strip()
        if not review:
            item["analysis"] = []
            continue

        print(f"\n🔍 第 {i+1} 筆評論分析中...")
        try:
            result = call_ollama(review)
            print(result)
            item["analysis"] = [
                {
                    "主題": r["topic"],
                    "對象": r["target"],
                    "構面": r["facet"],
                    "情感": r["sentiment"]
                } for r in result
            ]
        except Exception as e:
            print(f"⚠️ 分析失敗：{e}")
            item["analysis"] = []

        sleep(1)  # 加入 sleep 保險，不要太密集呼叫模型

    # 3. 輸出含分析結果的新檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 分析完成，結果已儲存至：{output_path}")


if __name__ == "__main__":
    input_file = "reviews_output.json"     # 你的原始評論 JSON 檔
    output_file = "analyzed_reviews.json"  # 分析後輸出的新檔案

    analyze_reviews_from_file(input_file, output_file)
