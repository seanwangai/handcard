import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import pandas as pd
import logging
import sys
import base64
from google.ai.generativelanguage_v1beta.types import content
import traceback
import time
import tempfile
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(page_title="PDF AI阅读助手", layout="wide")

# 初始化Gemini


def initialize_gemini():
    try:
        logger.info("开始初始化Gemini模型")
        api_keys = st.secrets["GOOGLE_API_KEYS"]
        if not api_keys:
            logger.error("API密钥未设置")
            st.error("请在secrets.toml中设置GOOGLE_API_KEYS")
            return None

        genai.configure(api_key=api_keys[0])
        logger.info("已配置API密钥")

        # 配置结构化输出
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_schema": genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "產品亮點": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "市场价格": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "直播价格": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "产品信息": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "口味": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "赠品": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "产品卖点": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                    "其他优势": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                    ),
                },
            ),
            "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config
        )
        logger.info("Gemini模型初始化成功")
        return model
    except Exception as e:
        logger.error(f"初始化Gemini时出错: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"初始化模型失败: {str(e)}")
        return None


def resize_image(image, max_size=1024):
    """调整图片大小，确保不超过API限制"""
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def convert_pdf_to_images(pdf_file):
    try:
        logger.info("开始转换PDF文件")
        images = []
        pdf_data = pdf_file.read()

        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        logger.info(f"PDF文件共 {pdf_document.page_count} 页")

        for page_num in range(pdf_document.page_count):
            logger.info(f"正在处理第 {page_num + 1} 页")
            page = pdf_document[page_num]
            # 降低分辨率以减小文件大小
            pix = page.get_pixmap(matrix=fitz.Matrix(
                150/72, 150/72))  # 降至150 DPI

            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            # 调整图片大小
            img = resize_image(img)
            images.append(img)
            logger.info(f"第 {page_num + 1} 页处理完成，图片大小: {img.size}")

        pdf_document.close()
        logger.info("PDF转换完成")
        return images
    except Exception as e:
        logger.error(f"转换PDF时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 处理单页查询


def convert_image_to_base64(image):
    """将PIL Image转换为base64字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def query_page(model, image, question, page_num):
    try:
        logger.info(f"开始分析第 {page_num} 页")

        # 将PIL Image保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file, format='PNG')
            tmp_file_path = tmp_file.name

        try:
            # 上传图片到Gemini
            logger.info(f"上传图片到Gemini - 第 {page_num} 页")
            uploaded_file = genai.upload_file(
                tmp_file_path, mime_type="image/png")
            logger.info(f"图片上传成功: {uploaded_file.uri}")

            # 创建聊天会话
            chat = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            uploaded_file,
                            "分析这个产品的信息",
                        ],
                    }
                ]
            )

            logger.info(f"发送请求到Gemini API - 第 {page_num} 页")
            response = chat.send_message("请提供产品分析结果")

            logger.info(f"收到Gemini API响应 - 第 {page_num} 页")
            logger.info(f"响应内容: {response.text}")

            # 尝试解析JSON响应
            try:
                # 清理响应文本
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1]
                if "```" in text:
                    text = text.split("```")[0]

                result = json.loads(text.strip())
                logger.info(f"成功解析JSON响应 - 第 {page_num} 页")
                return result
            except Exception as e:
                logger.error(f"JSON解析错误 - 第 {page_num} 页: {str(e)}")
                return {
                    "產品亮點": "",
                    "市场价格": "",
                    "直播价格": "",
                    "产品信息": response.text,
                    "口味": "",
                    "赠品": "",
                    "产品卖点": "",
                    "其他优势": ""
                }
        finally:
            # 清理临时文件
            os.unlink(tmp_file_path)

    except Exception as e:
        logger.error(f"查询页面时出错 - 第 {page_num} 页: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "產品亮點": "处理错误",
            "市场价格": "",
            "直播价格": "",
            "产品信息": str(e),
            "口味": "",
            "赠品": "",
            "产品卖点": "",
            "其他优势": ""
        }


def main():
    try:
        st.title("PDF AI阅读助手")
        logger.info("应用程序启动")

        # 初始化模型
        model = initialize_gemini()
        if not model:
            return

        # 文件上传
        uploaded_file = st.file_uploader("上传PDF文件", type=['pdf'])

        if uploaded_file:
            logger.info(f"收到上传文件: {uploaded_file.name}")
            # 转换PDF为图片
            with st.spinner('正在处理PDF文件...'):
                images = convert_pdf_to_images(uploaded_file)
                st.session_state['images'] = images
                logger.info(f"PDF处理完成，共 {len(images)} 页")
                st.success(f'成功处理PDF文件，共{len(images)}页')

        if 'images' in st.session_state:
            if st.button('分析产品信息'):
                logger.info("开始分析所有页面")
                st.write("### 分页答案")

                all_results = []

                for i, image in enumerate(st.session_state['images']):
                    with st.expander(f"第 {i+1} 页的回答"):
                        with st.spinner(f'正在分析第 {i+1} 页...'):
                            response = query_page(model, image, "分析产品信息", i+1)
                            response['页码'] = i + 1
                            all_results.append(response)
                            st.json(response)

                if all_results:
                    logger.info("开始生成CSV文件")
                    df = pd.DataFrame(all_results)
                    columns = ['页码', '產品亮點', '市场价格', '直播价格', '产品信息',
                               '口味', '赠品', '产品卖点', '其他优势']
                    df = df[columns]

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="下载CSV文件",
                        data=csv,
                        file_name="产品分析结果.csv",
                        mime="text/csv"
                    )
                    logger.info("CSV文件生成完成")

    except Exception as e:
        logger.error(f"主程序出错: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()
