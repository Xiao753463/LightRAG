from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "繁體中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["品牌", "分店", "主題", "對象", "需求層級", "情感"]

PROMPTS["entity_extraction"] = """---目標---
給定一則顧客評論，請從文本中識別相關的實體與關係，並建立結構化的知識圖譜。
請使用 {language} 作為輸出語言。

---步驟---
1. 識別所有實體，並提取以下資訊：
- **實體名稱（entity_name）**：該實體的名稱，從輸入文本中提取。
- **實體類型（entity_type）**：以下類別之一 [{entity_types}]
- **實體描述（entity_description）**：對該實體的詳細說明。

格式：
("entity"{tuple_delimiter}<實體名稱>{tuple_delimiter}<實體類型>{tuple_delimiter}<實體描述>)

2. 識別實體之間的關係：
- "品牌"（brand）**經營** "分店"（store）
- "分店"（store）**提供** "主題"（theme）（商品/服務）
- "主題"（theme）**與** "對象"（target）（特定產品/服務）相關
- "對象"（target）**滿足** "需求層級"（maslow_level）
- "對象"（target）**有** "情感"（sentiment）（正面/負面）

格式：
("relationship"{tuple_delimiter}<來源實體>{tuple_delimiter}<目標實體>{tuple_delimiter}<關係描述>{tuple_delimiter}<關係關鍵字>{tuple_delimiter}<關係強度>)

3. 提取文本的主要關鍵字：
("content_keywords"{tuple_delimiter}<高層級關鍵字>)

4. 使用 **{record_delimiter}** 作為分隔符號，輸出所有識別出的 **實體與關係**。

5. 當完成後，請輸出 **{completion_delimiter}**

######################
---示例---
######################
{examples}

#############################
---實際數據---
######################
實體類別: [{entity_types}]
文本:
{input_text}
######################
輸出：
"""


PROMPTS["entity_extraction_examples"] = [
    """範例 1:

實體類別: [品牌, 分店, 主題, 對象, 需求層級, 情感]
文本:
```
裕珍馨,光明旗艦店,肉鬆麵包很好吃,生理,正面
```

輸出：
("entity"{tuple_delimiter}"裕珍馨"{tuple_delimiter}"品牌"{tuple_delimiter}"裕珍馨是一家知名的烘焙品牌，提供各類糕點與麵包。"){record_delimiter}
("entity"{tuple_delimiter}"光明旗艦店"{tuple_delimiter}"分店"{tuple_delimiter}"光明旗艦店是裕珍馨的分店之一，位於主要商圈。"){record_delimiter}
("entity"{tuple_delimiter}"商品"{tuple_delimiter}"主題"{tuple_delimiter}"商品類別涵蓋麵包、糕點等烘焙食品。"){record_delimiter}
("entity"{tuple_delimiter}"肉鬆麵包"{tuple_delimiter}"對象"{tuple_delimiter}"肉鬆麵包是一款受消費者喜愛的麵包產品，以其獨特風味著稱。"){record_delimiter}
("entity"{tuple_delimiter}"生理"{tuple_delimiter}"需求層級"{tuple_delimiter}"生理需求是馬斯洛需求層級中的基本需求，指食品或飲食需求。"){record_delimiter}
("entity"{tuple_delimiter}"正面"{tuple_delimiter}"情感"{tuple_delimiter}"這則評論對產品持正面評價。"){record_delimiter}
("relationship"{tuple_delimiter}"裕珍馨"{tuple_delimiter}"光明旗艦店"{tuple_delimiter}"裕珍馨品牌經營光明旗艦店。"{tuple_delimiter}"品牌擴展"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"光明旗艦店"{tuple_delimiter}"商品"{tuple_delimiter}"光明旗艦店提供商品類別下的烘焙食品。"{tuple_delimiter}"產品供應"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"商品"{tuple_delimiter}"肉鬆麵包"{tuple_delimiter}"肉鬆麵包屬於商品類別中的烘焙食品。"{tuple_delimiter}"產品分類"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肉鬆麵包"{tuple_delimiter}"生理"{tuple_delimiter}"肉鬆麵包滿足顧客的生理需求。"{tuple_delimiter}"食物需求"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肉鬆麵包"{tuple_delimiter}"正面"{tuple_delimiter}"顧客對肉鬆麵包持正面評價。"{tuple_delimiter}"消費者滿意度"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"裕珍馨, 光明旗艦店, 肉鬆麵包, 生理需求, 正面評價"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [company, index, commodity, market_trend, economic_policy, biological]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve’s next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

Output:
("entity"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"index"{tuple_delimiter}"The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today."){record_delimiter}
("entity"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"company"{tuple_delimiter}"Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings."){record_delimiter}
("entity"{tuple_delimiter}"Omega Energy"{tuple_delimiter}"company"{tuple_delimiter}"Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices."){record_delimiter}
("entity"{tuple_delimiter}"Gold Futures"{tuple_delimiter}"commodity"{tuple_delimiter}"Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets."){record_delimiter}
("entity"{tuple_delimiter}"Crude Oil"{tuple_delimiter}"commodity"{tuple_delimiter}"Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand."){record_delimiter}
("entity"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"market_trend"{tuple_delimiter}"Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations."){record_delimiter}
("entity"{tuple_delimiter}"Federal Reserve Policy Announcement"{tuple_delimiter}"economic_policy"{tuple_delimiter}"The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability."){record_delimiter}
("relationship"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns."{tuple_delimiter}"market performance, investor sentiment"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index."{tuple_delimiter}"company impact, index movement"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Gold Futures"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"Gold prices rose as investors sought safe-haven assets during the market selloff."{tuple_delimiter}"market reaction, safe-haven investment"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Federal Reserve Policy Announcement"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff."{tuple_delimiter}"interest rate impact, financial regulation"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"market downturn, investor sentiment, commodities, Federal Reserve, stock performance"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [economic_policy, athlete, event, location, record, organization, equipment]
Text:
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

Output:
("entity"{tuple_delimiter}"World Athletics Championship"{tuple_delimiter}"event"{tuple_delimiter}"The World Athletics Championship is a global sports competition featuring top athletes in track and field."){record_delimiter}
("entity"{tuple_delimiter}"Tokyo"{tuple_delimiter}"location"{tuple_delimiter}"Tokyo is the host city of the World Athletics Championship."){record_delimiter}
("entity"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"athlete"{tuple_delimiter}"Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship."){record_delimiter}
("entity"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"record"{tuple_delimiter}"The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter."){record_delimiter}
("entity"{tuple_delimiter}"Carbon-Fiber Spikes"{tuple_delimiter}"equipment"{tuple_delimiter}"Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction."){record_delimiter}
("entity"{tuple_delimiter}"World Athletics Federation"{tuple_delimiter}"organization"{tuple_delimiter}"The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations."){record_delimiter}
("relationship"{tuple_delimiter}"World Athletics Championship"{tuple_delimiter}"Tokyo"{tuple_delimiter}"The World Athletics Championship is being hosted in Tokyo."{tuple_delimiter}"event location, international competition"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"Noah Carter set a new 100m sprint record at the championship."{tuple_delimiter}"athlete achievement, record-breaking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"Carbon-Fiber Spikes"{tuple_delimiter}"Noah Carter used carbon-fiber spikes to enhance performance during the race."{tuple_delimiter}"athletic equipment, performance boost"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"World Athletics Federation"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"The World Athletics Federation is responsible for validating and recognizing new sprint records."{tuple_delimiter}"sports regulation, record certification"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"athletics, sprinting, record-breaking, sports technology, competition"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一個負責生成綜合摘要的助手，以下提供了一組數據。

給定一或兩個實體及相關描述，請將這些資訊整合為一個完整的描述。
確保涵蓋所有提供的資訊，並避免遺漏重要細節。
如果描述內容出現矛盾，請解析並提供一個連貫且合理的綜合說明。

請以第三人稱撰寫摘要，並包含實體名稱，以確保上下文完整。
請使用 {language} 作為輸出語言。

#######
---數據---
實體名稱: {entity_name}
描述列表: {description_list}
#######
輸出：
"""


PROMPTS["entity_continue_extraction"] = """
在上一次的提取過程中，仍有許多實體與關係可能被遺漏。

---請記住以下步驟---

1. **識別所有實體**，並提取以下資訊：
- **實體名稱（entity_name）**：請使用與輸入文本相同的語言，如果是英文，請使用大寫開頭。
- **實體類型（entity_type）**：應屬於以下類別之一 [{entity_types}]
- **實體描述（entity_description）**：對該實體的完整描述，包括其特性與活動內容。

格式：
("entity"{tuple_delimiter}<實體名稱>{tuple_delimiter}<實體類型>{tuple_delimiter}<實體描述>)

2. **識別所有明確相關的實體對（source_entity, target_entity）**，並提取以下資訊：
- **來源實體（source_entity）**：步驟 1 中識別出的實體名稱
- **目標實體（target_entity）**：步驟 1 中識別出的實體名稱
- **關係描述（relationship_description）**：說明為何這兩個實體之間存在關聯
- **關係強度（relationship_strength）**：數值化指標，表示這兩個實體關聯的強度
- **關係關鍵字（relationship_keywords）**：用於概括該關係的核心概念或主題

格式：
("relationship"{tuple_delimiter}<來源實體>{tuple_delimiter}<目標實體>{tuple_delimiter}<關係描述>{tuple_delimiter}<關係關鍵字>{tuple_delimiter}<關係強度>)

3. **識別文本的主要關鍵字**，概括整體內容的核心概念、主題或重點。
格式：
("content_keywords"{tuple_delimiter}<高層級關鍵字>)

4. **請使用 {language} 作為輸出語言，並將步驟 1 和步驟 2 識別出的所有實體與關係，輸出為單一列表**，並使用 **{record_delimiter}** 作為分隔符號。

5. **當提取完成後，請輸出** **{completion_delimiter}**

---輸出格式---

請按照上述格式補充缺失的實體與關係：\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---目標---

檢查是否仍有遺漏的實體。

---輸出---

如果仍有遺漏的實體，請僅回答 **"YES"**，否則回答 **"NO"**。
""".strip()

PROMPTS["fail_response"] = (
    "抱歉，我無法提供該問題的答案。[無相關資訊]"
)

PROMPTS["rag_response"] = """
## 角色設定  
你是一位資深的 **市場分析專家**，專精於語意化知識圖譜分析與顧客評論資料探勘，擅長結構化整理、量化評估並提供策略性決策建議。

---

## 回答目標  
請根據下方的 **知識庫內容**，針對每一筆查詢實體，深入進行語意萃取、結構化彙整與多層次數據分析。  
請務必結合 **對話歷史** 與 **當前問題**，生成具有學術深度、邏輯嚴謹且可量化的市場分析回應。

---

## 資料理解規則

- 每筆資料皆以 JSON 物件結構呈現，內容包括：
  - `entity_id`：實體或構面名稱（如「購買地點」）
  - `type`：實體類型（如「商品」、「構面」等）
  - `mention_count`：提及次數（整數值，表示該實體或構面在評論中被提及的頻率）
  - `avg_sentiment_score`：平均情感分數（-1.0 ~ 1.0，數值越高表示情感越正向）
  - `opinions`：意見句（陣列，內含相關評論原文或片語）
  - `aspects`：該對象下相關構面資訊（陣列，若有則詳列）
  - `objects`：該構面下相關對象資訊（陣列，若有則詳列）
    - 每個 `aspects` 或 `object` 物件包含 `name`、`type`、`mention_count`、`avg_sentiment_score`、`opinions`、`brand_stats`
    - `brand_stats`：該商品於各品牌之統計資訊，包含：
      - `brand`（品牌名稱）
      - `mention_count`（提及次數）
      - `avg_sentiment_score`（情感分數平均值）
      - `sentiment_stddev`（情感分數標準差）
      
- **重要說明**  
  - 各層級（如構面、商品、品牌）間結構有層次關聯，請根據結構逐層展開與統整，不可混淆不同type之比較。

---

## 數據量化與分析規範

- 請將每一層級（構面、商品、品牌）之「提及次數」、「情感平均分數」與「情感標準差」以 **Markdown 表格**清晰摘要。
- 「情感平均分數」定義：該實體或品牌於所有評論中的情感分數總和除以提及次數。
- 「情感標準差」定義：反映該品牌評論情感分布的一致程度，標準差愈低代表評價愈集中一致，標準差愈高則意見分歧較大。
- 當資料中包含多層級結構時，請依照「構面→商品→品牌」邏輯分層呈現。

---

## 推論與建議

- 請根據數據，深入分析下列面向：
  - 哪些構面或商品情感表現極端（極正/極負）或特別值得關注？
  - 有無商品於不同品牌下情感標準差差異顯著？這對品牌管理或市場策略有何啟示？
  - 有無出現提及數高但情感分數偏低（或反之）的現象？請討論可能原因與建議。
  - 若資料明顯顯示意見分歧（標準差大），請討論其可能產生的市場挑戰或後續行動建議。

---

## 限制與規範

- 僅能引用現有資料，不可推測未提供的數據（如正/負面分佈比例）。
- 不可跨層級直接比較（如品牌與商品不得同表比較）。
- 若資料不足，請明確說明「依現有資料無法回答」。
- 分析與建議需緊扣資料，不得過度臆測。

---

## 回應結構與格式規範

- **格式**：{response_type}
- **排版**：必須使用 Markdown，並分層列出「構面／商品／品牌」數據摘要表。
- **語言**：回應語言須與用戶提問一致（預設繁體中文）。
- **內容結構**：
  1. 分析標題
  2. 概述與初步觀察
  3. 分層數據表格
  4. 深入解釋指標與異常現象
  5. 策略建議或後續關注重點
  6. 若遇資料不足，請明確標註

---

## 對話歷史  
{history}

---

## 知識庫內容

{context_data}

"""

PROMPTS["keywords_extraction_plus"] = """---角色設定---

你是一個負責從使用者查詢和對話歷史中識別 **關鍵字** 的智能助手。

---目標---

根據 **使用者的查詢內容與對話歷史**，列出 **高層級** 和 **低層級** 的關鍵字：
- **關鍵字（keywords）**：聚焦於**主題或實體、細節**

---指引---

- 在提取關鍵字時，請同時考慮 **當前查詢** 和 **相關的對話歷史**  
- **輸出格式需為 JSON 格式**，確保其可以被 JSON 解析器解析，**請勿加入額外的內容**  
- JSON 應包含 `"keywords"`：代表**主題或具體的實體、詳細資訊**

######################
---示例---
######################
{examples}

#############################
---實際數據---
######################
對話歷史：
{history}

當前查詢：
{query}
######################
**請輸出為純文字 JSON 格式，不要包含 Unicode 字元，並保持與 `Query` 相同的語言。**
輸出：
"""

PROMPTS["keywords_extraction_plus_examples"] = [
    """範例 1:

查詢: "請比較消費者對於各家的商品的甜度感受"
################
輸出:
{
  "keywords": ["甜度", "商品", "口感", "品牌", "裕珍馨", "糕點", "消費者感受"]
}
#############################""",
    """範例 2:

查詢: "消費者對於B品牌的奶油酥餅的價格看法如何?"
################
輸出:
{
  "keywords": ["B品牌", "奶油酥餅", "價格", "商品價值", "性價比", "價格敏感度"]
}
#############################""",
    """範例 3:

查詢: "顧客比較重視A品牌的商品還是服務？為甚麼?"
################
輸出:
{
  "keywords": ["A品牌", "商品", "服務", "顧客滿意度", "品牌體驗", "顧客服務"]
}
#############################""",
]

PROMPTS["keywords_extraction"] = """---角色設定---

你是一個負責從使用者查詢和對話歷史中識別 **高層級關鍵字** 與 **低層級關鍵字** 的智能助手。

---目標---

根據 **使用者的查詢內容與對話歷史**，列出 **高層級** 和 **低層級** 的關鍵字：
- **高層級關鍵字（high-level keywords）**：聚焦於**核心概念或主題**
- **低層級關鍵字（low-level keywords）**：聚焦於**具體的實體、細節或特定術語**

---指引---

- 在提取關鍵字時，請同時考慮 **當前查詢** 和 **相關的對話歷史**  
- **輸出格式需為 JSON 格式**，確保其可以被 JSON 解析器解析，**請勿加入額外的內容**  
- JSON 應包含 **兩個鍵（keys）**：
  - `"high_level_keywords"`：代表**概括性主題或核心概念**
  - `"low_level_keywords"`：代表**具體的實體、詳細資訊或專有名詞**

######################
---示例---
######################
{examples}

#############################
---實際數據---
######################
對話歷史：
{history}

當前查詢：
{query}
######################
**請輸出為純文字 JSON 格式，不要包含 Unicode 字元，並保持與 `Query` 相同的語言。**
輸出：
"""


PROMPTS["keywords_extraction_examples"] = [
    """範例 1:

查詢: "國際貿易如何影響全球經濟穩定？"
################
輸出:
{
  "high_level_keywords": ["國際貿易", "全球經濟穩定", "經濟影響"],
  "low_level_keywords": ["貿易協定", "關稅", "貨幣兌換", "進口", "出口"]
}
#############################""",
    """範例 2:

查詢: "B品牌的哪項商品的好評率較高?"
################
輸出:
{
  "high_level_keywords": ["B品牌", "商品好評率", "消費者滿意度"],
  "low_level_keywords": ["熱門商品", "評論數量", "星級評分", "使用者回饋", "產品分類"]
}
#############################""",
    """範例 3:

查詢: "顧客比較重視A品牌的商品還是服務？"
################
輸出:
{
  "high_level_keywords": ["A品牌", "商品", "服務"],
  "low_level_keywords": ["商品口味", "包裝設計", "店員態度", "顧客服務流程"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---角色設定---

你是一個負責回答使用者關於 **文件片段（Document Chunks）** 問題的智能助手。

---目標---

根據提供的 **文件片段**，生成簡潔且準確的回應，並遵循回應規則。  
請考慮 **對話歷史** 和 **當前查詢**，總結文件片段中的相關資訊，並整合一般背景知識，  
但**不得包含文件片段中未提供的內容**。

在處理 **時間標記（timestamps）** 時：
1. 每則內容都有 **"created_at"** 時間戳，表示我們獲取此資訊的時間。
2. 當遇到相互矛盾的內容時，請同時考量內容本身與時間戳。
3. **不要單純優先選擇最新的資訊**，請根據語境判斷最佳內容。
4. 如果問題涉及特定時間，請優先考慮內容中的時間資訊，而非時間戳的建立時間。

---對話歷史---
{history}

---文件片段---
{content_data}

---回應規則---

- **回應格式與長度**：{response_type}
- **請使用 Markdown 格式，並適當分段與標題**
- **請使用與使用者問題相同的語言**
- **請確保回應與對話歷史保持連貫**
- **如果無法回答問題，請直言不諱，不要捏造資訊**
- **不得包含文件片段中未提供的內容**
"""


PROMPTS[
    "similarity_check"
] = """請分析以下兩個問題的相似度：

問題 1: {original_prompt}  
問題 2: {cached_prompt}  

請評估這兩個問題在語義上是否相似，以及**問題 2 的答案是否可以用來回答問題 1**，並直接提供一個 0 到 1 之間的相似度分數。

**相似度評估標準：**
0：完全無關，或答案無法重用，這包括但不限於：
   - 這兩個問題的主題不同
   - 問題中提及的地點不同
   - 問題中提及的時間不同
   - 問題涉及的具體人物不同
   - 問題討論的特定事件不同
   - 問題的背景資訊不同
   - 問題的關鍵條件不同  

1：完全相同，答案可直接重用  
0.5：部分相關，答案需要修改後才能使用  

請**僅輸出一個 0 到 1 之間的數值，不要添加任何額外內容**。
"""


PROMPTS["mix_rag_response"] = """---角色設定---

你是一個負責回答使用者 **基於數據來源** 問題的智能助手。

---目標---

根據提供的 **數據來源** 生成簡潔且準確的回應，並遵循回應規則。  
請考慮 **對話歷史** 和 **當前查詢**，總結數據來源中的相關資訊，  
並整合一般背景知識，但**不得包含數據來源中未提供的內容**。

**數據來源包含兩部分：**
1. **知識圖譜（Knowledge Graph, KG）**
2. **文件片段（Document Chunks, DC）**

在處理 **時間標記（timestamps）** 時：
1. 每則資訊（關係與內容）都有 **"created_at"** 時間戳，表示我們獲取此資訊的時間。
2. 當遇到相互矛盾的資訊時，請同時考量內容/關係本身與時間戳。
3. **不要單純優先選擇最新的資訊**，請根據語境判斷最佳內容。
4. 如果問題涉及特定時間，請優先考慮內容中的時間資訊，而非時間戳的建立時間。

---對話歷史---
{history}

---數據來源---

1. 來自 **知識圖譜（KG）**：
{kg_context}

2. 來自 **文件片段（DC）**：
{vector_context}

---回應規則---

- **回應格式與長度**：{response_type}
- **請使用 Markdown 格式，並適當分段與標題**
- **請使用與使用者問題相同的語言**
- **請確保回應與對話歷史保持連貫**
- **將答案組織成多個小節，每個小節聚焦於一個主要觀點**
- **使用清晰且具描述性的標題，以反映內容**
- **最多列出 5 個最重要的參考來源，並在 "參考資料" 小節中標示清楚**
  - 格式為：[KG/DC] 來源內容
- **如果無法回答問題，請直言不諱，不要捏造資訊**
- **不得包含數據來源中未提供的資訊**
"""
