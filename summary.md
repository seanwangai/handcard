用 streamlit 做一個 檔案AI閱讀助手

使用gemini，key 我會放在 secrets.toml的 GOOGLE_API_KEYS = [] 內
範例code 如下
"""
$ pip install google.ai.generativelanguage

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_schema": content.Schema(
    type = content.Type.OBJECT,
    properties = {
      "产品名称": content.Schema(
        type = content.Type.STRING,
      ),
      "市场价格 ": content.Schema(
        type = content.Type.STRING,
      ),
      "直播价格": content.Schema(
        type = content.Type.STRING,
      ),
      "产品信息": content.Schema(
        type = content.Type.STRING,
      ),
      "口味": content.Schema(
        type = content.Type.STRING,
      ),
      "赠品": content.Schema(
        type = content.Type.STRING,
      ),
      "产品卖点": content.Schema(
        type = content.Type.STRING,
      ),
      "其他优势": content.Schema(
        type = content.Type.STRING,
      ),
    },
  ),
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-lite-preview-02-05",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)
"""


目標是
使用者上傳 PDF 後
一頁一頁 先把 pdf 轉乘 圖片，然後把圖片 一頁一頁丟給gemini llm 

使用者可以輸入問題 系統會分別把每一頁 和問題 丟到llm 分別回答 




{
  "type": "object",
  "properties": {
    "產品亮點": {
      "type": "string"
    },
    "市场价格": {
      "type": "string"
    },
    "直播价格": {
      "type": "string"
    },
    "产品信息": {
      "type": "string"
    },
    "口味": {
      "type": "string"
    },
    "其他优势": {
      "type": "string"
    }
  }
}