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


prompt = """
輸出成 csv 格式, 要有 產品亮點 市场价格 直播价格 产品信息 口味 贈品 产品卖点 其他优势 欄位
注意 產品亮點 要學習範例內的格式，要有【】
以下是 範例 
{
"產品亮點": {
      "大家是不是觉得最近夏日炎炎，特别想吹空调，喝冷饮，吃东西偶尔觉得不是特别有胃口但深夜总想吃点重口味的呢，你知道为什么吗？因为三伏天来了，三伏天，正是我们进行数字管理的黄金时期，在这个阶段内，你的努力很有可能事半功倍，但是如果错过了，就要再等一年。今天的这款薏米水和桂圆水特别适合正在数字管理的宝贝们，大家一定要抓住入伏40天
【试用人群】
1. 早上起来脸泡、手泡、脚泡的，感觉怎么自己的脸照镜子和昨晚都不太一样的，一定要早上来一杯黑咖啡的  2. 别人醒来一秒钟脉动进入工作状态的，而你浑身懒洋洋的，容易累、容易乏的  3. 最近爱吹空调、吃冷饮、喝冰水冰咖啡并且还管不住嘴，爱吃油的辣的 4. 大便挂壁的，背上脸上点点的
【配料表】0糖0脂0卡路里0添加0碳水，微沸慢熬，早上的第一杯放心水，冷热皆宜随时随地打开就可以直接喝，平时常温储存
【五指毛桃薏米水】就是被称为"广东人参"的五指毛桃+晾晒10天以上的薏米，选用炒过的薏米，因为这样不仅更给力，而且更好喝，每瓶添加超过750mg的五指毛桃，以及5000mg薏米
【枸杞桂圆水】精选桂圆三宝："桂圆肉、桂圆核、桂圆壳"，每瓶添加3500mg以上的桂圆干，经过48h烘烤，碾碎，整颗桂圆连肉带壳一起煮，营养不流失，再加500mg+的NFC枸杞原浆
陈皮水：水、陈皮、白茶、青柑皮浓缩液、碳酸氢钠
喝法：1.直接喝（冷热都好喝）2.搭配咖啡成为中式茶咖  3. 柠檬片或柠檬汁。4.炖鸡汤排骨
    },
    "市场价格": {
      "39.9"
    },
    "直播价格": {
      "19.9"
    },
    "产品信息": {
      "500ml*5瓶"
    },
    "口味": {
      "薏米水/陈皮水/桂圆水"
    },
    "贈品": {
      "买4瓶送1瓶"
    }
}


"""


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
                            prompt,
                        ],
                    }
                ]
            )

            logger.info(f"发送请求到Gemini API - 第 {page_num} 页")
            response = chat.send_message("请提供产品分析结果")
            logger.info(f"收到Gemini API响应 - 第 {page_num} 页")
            logger.info(response.text)
            return response.text  # 直接返回响应文本

        finally:
            # 清理临时文件
            os.unlink(tmp_file_path)

    except Exception as e:
        logger.error(f"查询页面时出错 - 第 {page_num} 页: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def main():
    try:
        st.title("PDF AI阅读助手")
        logger.info("应用程序启动")

        # 初始化 session state
        if 'all_results' not in st.session_state:
            st.session_state.all_results = []
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = None
        if 'current_file_name' not in st.session_state:
            st.session_state.current_file_name = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False

        # 初始化模型
        model = initialize_gemini()
        if not model:
            return

        # 文件上传
        uploaded_file = st.file_uploader("上传PDF文件", type=['pdf'])

        if uploaded_file:
            # 检查是否需要重新处理PDF
            if (st.session_state.current_file_name != uploaded_file.name or
                    st.session_state.processed_images is None):

                logger.info(f"处理新文件: {uploaded_file.name}")
                # 转换PDF为图片
                with st.spinner('正在处理文件...'):
                    images = convert_pdf_to_images(uploaded_file)
                    st.session_state.processed_images = images
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.processing_complete = False
                    st.session_state.all_results = []  # 清空之前的结果

                    # 分析每一页
                    for i, image in enumerate(images):
                        with st.spinner(f'正在分析第 {i+1}/{len(images)} 页...'):
                            try:
                                response = query_page(
                                    model, image, "分析产品信息", i+1)
                                if response:
                                    result = json.loads(response)
                                    result['页码'] = i + 1
                                    st.session_state.all_results.append(result)
                            except Exception as e:
                                logger.error(f"分析第 {i+1} 页时出错: {str(e)}")
                                st.error(f"第 {i+1} 页分析失败，继续处理其他页面")
                                continue

                    st.session_state.processing_complete = True
                    st.success(f'成功处理并分析完成，共{len(images)}页')

            # 显示结果（无论是新处理的还是之前处理过的）
            if st.session_state.processing_complete:
                st.write("### 分析结果")

                # 显示每页的结果
                for result in st.session_state.all_results:
                    with st.expander(f"第 {result['页码']} 页的分析", expanded=True):
                        st.json(result)

                # 生成CSV下载按钮
                if st.session_state.all_results:
                    df = pd.DataFrame(st.session_state.all_results)

                    # 预定义所有可能的列
                    expected_columns = [
                        '页码',
                        '產品亮點',
                        '市场价格',
                        '直播价格',
                        '产品信息',
                        '口味',
                        '赠品',
                        '产品卖点',
                        '其他优势'
                    ]

                    # 确保所有列都存在，缺失的填充空字符串
                    for col in expected_columns:
                        if col not in df.columns:
                            df[col] = ''

                    # 按预定义顺序排列列
                    df = df[expected_columns]

                    # 将所有 NaN 值替换为空字符串
                    df = df.fillna('')

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="下载CSV文件",
                        data=csv,
                        file_name=f"{st.session_state.current_file_name}_分析结果.csv",
                        mime="text/csv",
                        key="download_button"
                    )

    except Exception as e:
        logger.error(f"主程序出错: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()
