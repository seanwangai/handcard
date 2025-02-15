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
            "max_output_tokens": 81920,
            # "response_schema": genai.protos.Schema(
            #     type=genai.protos.Type.OBJECT,
            #     properties={
            #         "產品亮點": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "市场价格": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "直播价格": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "产品信息": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "口味": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "赠品": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "产品卖点": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #         "其他优势": genai.protos.Schema(
            #             type=genai.protos.Type.STRING,
            #         ),
            #     },
            # ),
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
            logger.info(f"正在轉換第 {page_num + 1} 页")
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
把圖片中的信息，分別填入以下欄位: 产品名称 产品卖点 市场价格 直播价格 产品信息 其他优势，並以json 格式輸出


======== 格式注意事項：
json 維持一層就好 key: string value，確保都是 "key": "value"
只能有一個"value" string，像是下面這樣
"市场价格": "无刺绣|小号(无拉链):39.9\n无刺绣|大号(拉链款):59.9",

不要出現 太多 \ n 換行 的bug，最多一個 \ n

以下是 範例 一定要依照這個格式輸出，注意 開頭要是 { 結尾要是 }，不能是[ 或是 ]
{
'产品名称': "",
"产品卖点": "大家是不是觉得最近夏日炎炎，........",
    "市场价格":  "39.9",
    "直播价格": "19.9",
    "产品信息": "500ml*5瓶",
    "其他优势": ""
}

＝＝＝＝＝ 欄位注意事項：
其他优势： 1. 寫一首以這個產品的特點為主題的古詩，至少8句以上，每個句子7個字，要押韻，要唯美有創意，

＝＝＝＝＝ 产品卖点 注意事項

产品卖点 格式注意事項：
1. 要改寫，模仿以下幾個範例的寫法，不要只貼原本的文字，要模仿範例中的語氣和寫作方式
2. 寫的越詳細越好，越長越完整越好，字數越多越好，如果圖片內容太少可以發揮想像力寫多一點
4. 如果是食品 也適度納入以下的資訊
  - 食品成分：配料表是否干净，有无添加剂、蔗糖、防腐剂、反式脂肪、香精等。
  - 食用方式及食用场景。
5. 产品卖点 內文列點的時候 1 2 3 4 5 改用 emoji 1️⃣2️⃣3️⃣4️⃣5️⃣

＝＝产品卖点 食品範例
分【引入段落】、【配料表】、【卖点（好原料，好工艺，对比）】、【吃法】和【人群】这几个部分，文案需要包含以下几个部分，并模仿范例文案的写作方式：

**1. 引入段落 (模仿范例的【引入】):**
   - 针对特定人群 (例如女性)，点出她们的常见痛点或需求。
   - 突出产品与这些痛点或需求的关联性。
   - 使用生动形象的比喻或口语化的表达，吸引读者。
   - 产品定位要清晰 (例如代餐、零食、饮品等)。

**2. 配料表 (模仿范例的【配料表】):**
   - 列出产品的核心成分，强调健康、天然、无添加等特点 (如果适用)。
   - 用简洁明了的语言呈现配料信息。
   - 可以用对比或数据 (例如热量) 来突出产品的优势。

**3. 卖点 (模仿范例的【卖点】):**
   - 从 "好原料"、"好工艺"、"品牌/同款对比" 等角度，多维度呈现产品卖点。
   - 每个卖点都要具体展开，解释 "好" 在哪里，能给用户带来什么好处。
   - 语言要简洁有力，突出产品的核心优势。

**4. 吃法/用法 (模仿范例的【吃法】):**
   - 提供多种食用或使用场景建议，增加产品的实用性和趣味性。
   - 可以包括 "直接吃/用"、"DIY 搭配"、"创意吃法/用法" 等。
   - 场景描述要具体，让用户容易想象。

**5. 人群 (模仿范例的【人群】):**
   - 明确产品的目标人群，可以细化到不同年龄、职业、生活习惯等。
   - 针对不同人群的特点，描述产品对他们的价值和意义。
   - 可以使用 "也适合..." 等语句扩展人群范围。

**参考文案:**
引入:一款超适合女生的陈皮莲子红豆沙，面黄面暗、姨妈不顺的女孩子不仅可以用来做代餐，还可以作为宝藏甜品，女生应该懂红豆的作用吧？平时需要涂腮红的，它就是很好的腮红神器
【配料表】0脂0防腐剂0增稠剂，红豆，莲子，陈皮，冰糖，配料表干净，热量180卡=1.5个苹果
【卖点】
1. 好原料：精选去皮红豆，取精华，不会有苦感，而且还额外搭配陈皮莲子，调理脾胃，养颜面色好
2. 好工艺：水洗沙工艺，先蒸再滤最后洗，吃到嘴里绵绵沙沙的，口感特别好
3. 线下商超山某姆同款，而且配料、含量、大小都要更好吃法：1.直接吃（冷热都好吃，直接可以撕开铝纸就可以用微波炉加热）喜欢diy的搭配热牛奶2.新中式红豆沙奶茶  3. 自制红豆沙三明治吐司/馒头，小朋友都很爱吃人群:小孩子不爱吃饭经常吃零食的，长期点外卖的，健身轻体控制身材的，宿舍党，上班族老人孕妇小孩都可以

＝＝产品卖点 日用品範例
按照【个人体验引入】、【痛点+效果展示】、【适用人群和场景】、【产品推荐和优势】、【产品原理和功效（可选）】、【产品卖点细化（材质、设计、寓意等）】这几个部分，写一篇产品文案。

**文案风格要求:**
- 以第一人称 "我" 开头，分享个人体验或感受
- 结合传统文化或养生理念（如果产品相关）
- 语言口语化、亲切自然，像朋友推荐好物
- 突出产品的功能性和带来的实际好处
- 融入一些情感价值或文化寓意（如果产品适合）

**参考文案:**
我从古法健身开始接触到咱们传统白大褂，就是从八段锦和拍八虚开始，尤其是拍八虚，对我得帮助特别大，只需轻轻拍打身体的8个部位，就可以让我们从早上开始感受到精气神，并且缓解疲劳。无论是日常调理、上班族久坐后的腰酸背痛，还是熬夜族的头晕乏力，都去拍拍八虚，那拍八虚我推荐使用长柄经络拍，拿起省力、又能拍到你拍不到得地方、比锤要更模拟出空心掌的感觉
【拍打原因】：1. 疏某通经络：经络是人体气运行的通道，经络通了，身体才能健康。经络拍通过轻柔的拍打，帮助疏某通经络，让气血顺畅运行。2.  日常调理身体的好搭子：比如在艾灸前用经络拍疏通经络，能让艾灸的效果更深入；在按摩后用经络拍巩固，都有事半功倍的体会 3. 肌肉放松：久坐久站、运动、开车长时间1个动作，长时间的劳累会让身体积累大量乳酸，经络拍可以促进红色液体循环，帮助身体排出乳酸，工作间隙放松，旅途劳顿缓解疲劳，居家按摩很好的工具。
【卖点】材质优选：绒布+弹性EVA材质，拍打舒适不易变形，很紧实皮肤接触到也没有觉得不舒服；大号还特别添加棉麻布+艾草（only），艾草性温，除了孕妈妈不要用，其他都有祛湿散寒的作用，大号可拆卸布套水洗，小号可直接水洗晒干
推荐大号小号一起入手：大号适合居家按摩，小号方便外出携带。
比市面上的很多经络拍要美观，柿柿如意的美好寓意，送礼佳品：柿柿如意经络拍的柿子刺绣图案取自《食材草本》的柿子原画，寓意吉祥，外包装带有提手的礼盒，很适合送礼。

"""

prompt = """
把圖片中的信息，填入以下欄位: 产品名称 爆款標題 中國古詩創作，並以json 格式輸出


== == == == 格式注意事項：
json 維持一層就好 key: string value，確保都是 "key": "value"
只能有一個"value" string，像是下面這樣
"爆款標題": "新手必看！ 后悔没早点刮肝经！保姆级教程，一学就会！",

不要出現 太多 \ n 換行 的bug，最多一個 \ n

以下是 範例 一定要依照這個格式輸出，注意 開頭要是 { 結尾要是 }，不能是[ 或是 ]
{
'产品名称': "",
"爆款標題": "1. 后悔没早点吃！ 亚麻籽奇亚籽燕麦脆片，健康美味，低卡饱腹！\n2. 救命！ 我真的爱上这款脆片了！营养丰富，口感酥脆，减肥必备！",
"中國古詩創作":  "",
}

＝＝＝＝＝ 爆款標題 注意事項：
生成10個爆款標題，學習範例的格式改寫

爆款標題範例：
新手必看！ 后悔没早点刮肝经！保姆级教程，一学就会！
救命！ 我真的会刮肝经了！疏肝解郁，告别emo体质！
女生一定要看！ 学会正确刮肝经，内调外养，格局炸裂！
答应我！ 一定要试试这个刮肝经的方法，绝了！
听我一句！ 刮肝经的最佳时间和禁忌，别再瞎刮了！
封神！ 自媒体养生博主私藏的刮肝经秘籍，直接抄作业！
千万不要！ 错过这个从0到1的刮肝经教程，让你少走弯路！
一出门就被问！ 坚持刮肝经，气色好了，人都自信了！
天花板！ 这才是刮肝经的正确打开方式，效果翻倍！
从0到1！ 自媒体养生博主教你如何正确刮肝经，附肝经位置和重点穴位！
格局炸裂！ 原来刮肝经还有这么多讲究，看完这篇你就明白了！
错过后悔！ 90%的人都不知道的刮肝经技巧，让你事半功倍！
无限回购！ 坚持刮肝经，身体越来越好，谁刮谁知道！
被追着问！ 每天刮一刮肝经，睡眠好了，皮肤也变好了！
超厉害！ 零基础也能学会的刮肝经手法，简单易学，效果惊人！
零失败！ 跟着我刮肝经，告别肝气郁结，做个快乐的养生达人！

＝＝＝＝＝ 中國古詩創作 注意事項
寫一首以這個產品的特點為主題的古詩，李白的風格，至少8句以上，每個句子剛好7個字，最後一個字要押韻 最後一個字一定要押韻

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
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0  # 当前处理的页码

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
                    st.session_state.current_page = 0  # 重置当前页码

            # 处理每一页
            if st.session_state.processed_images is not None:
                total_pages = len(st.session_state.processed_images)

                # 处理未完成的页面
                while st.session_state.current_page < total_pages:
                    image = st.session_state.processed_images[st.session_state.current_page]
                    with st.spinner(f'正在分析第 {st.session_state.current_page + 1}/{total_pages} 页...'):
                        try:
                            response = query_page(
                                model, image, "分析产品信息", st.session_state.current_page + 1)
                            if response:
                                result = json.loads(response)
                                result['页码'] = st.session_state.current_page + 1
                                st.session_state.all_results.append(result)
                                st.session_state.current_page += 1  # 处理下一页
                        except Exception as e:
                            logger.error(
                                f"分析第 {st.session_state.current_page + 1} 页时出错: {str(e)}")
                            st.error(
                                f"第 {st.session_state.current_page + 1} 页分析失败，继续处理其他页面")
                            st.session_state.current_page += 1  # 处理下一页

                # 检查是否处理完成
                if st.session_state.current_page >= total_pages:
                    st.session_state.processing_complete = True
                    st.success(f'成功处理并分析完成，共{total_pages}页')

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
                        '产品名称',
                        '产品卖点',
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
