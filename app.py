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
把圖片中的信息，分別填入以下欄位: 产品名称 产品卖点 市场价格 直播价格 产品信息 口味 贈品 其他优势，並以json 格式輸出


======== 格式注意事項：
json 維持一層就好 key: string value，確保都是 "key": "value"
只能有一個"value" string，像是下面這樣
"市场价格": "无刺绣|小号(无拉链):39.9\n无刺绣|大号(拉链款):59.9",

不要出現 太多 \ n 換行 的bug，最多一個 \ n


＝＝＝＝＝ 欄位注意事項：
 
产品卖点 格式注意事項：
要改寫不要只貼原本的文字，改寫的寫法如下：
"产品卖点 要模仿以下範例的寫法，記得全部都要改寫成以下的模式，要改寫不要只貼原本的文字，要改寫成以下的寫法：
大家是不是觉得最近夏日炎炎，特别想吹空调，喝冷饮，吃东西偶尔觉得不是特别有胃口但深夜总想吃点重口味的呢，你知道为什么吗？因为三伏天来了，三伏天，正是我们进行数字管理的黄金时期，在这个阶段内，你的努力很有可能事半功倍，但是如果错过了，就要再等一年。今天的这款薏米水和桂圆水特别适合正在数字管理的宝贝们，大家一定要抓住入伏40天
【试用人群】
1. 早上起来脸泡、手泡、脚泡的，感觉怎么自己的脸照镜子和昨晚都不太一样的，一定要早上来一杯黑咖啡的  2. 别人醒来一秒钟脉动进入工作状态的，而你浑身懒洋洋的，容易累、容易乏的  3. 最近爱吹空调、吃冷饮、喝冰水冰咖啡并且还管不住嘴，爱吃油的辣的 4. 大便挂壁的，背上脸上点点的
【配料表】0糖0脂0卡路里0添加0碳水，微沸慢熬，早上的第一杯放心水，冷热皆宜随时随地打开就可以直接喝，平时常温储存
【五指毛桃薏米水】就是被称为"广东人参"的五指毛桃+晾晒10天以上的薏米，选用炒过的薏米，因为这样不仅更给力，而且更好喝，每瓶添加超过750mg的五指毛桃，以及5000mg薏米
【枸杞桂圆水】精选桂圆三宝："桂圆肉、桂圆核、桂圆壳"，每瓶添加3500mg以上的桂圆干，经过48h烘烤，碾碎，整颗桂圆连肉带壳一起煮，营养不流失，再加500mg+的NFC枸杞原浆
陈皮水：水、陈皮、白茶、青柑皮浓缩液、碳酸氢钠。喝法：1.直接喝（冷热都好喝）2.搭配咖啡成为中式茶咖  3. 柠檬片或柠檬汁。4.炖鸡汤排骨"

所以 产品卖点 就是要有【】，要越詳細越好，越長越完整越好 越長越好 圖片內的文字有提到的 全部都要納入改寫
第一段都要有一段話術作為開頭，後面再把產品亮點列出
 還要適度納入以下的資訊
  - 成分：配料表是否干净，有无添加剂、蔗糖、防腐剂、反式脂肪、香精等。
  - 产品描述：产品独特卖点或其他优势。
  - 食用方式及食用场景。 
  - 营销话术
 -  禁忌人群

=== 

其他优势（现货、发货时效、物流、不发货地区、适用场景、门店等） 交給你自由發揮，要辨識圖片內有趣的

赠品 如果沒有 就顯示 "无" 

＝＝＝＝＝
內文列點的時候 不用換行 可以用 1️⃣2️⃣3️⃣4️⃣5️⃣ 
正確輸出格式：【使用场景】 1️⃣ 居家放松时 2️⃣工作间隙 3️⃣旅行途中
錯誤輸出格式：【使用场景】
1. 居家放松时
2. 工作间隙
3. 旅行途中

====
以下是 範例 一定要依照這個格式輸出，注意 開頭要是 { 結尾要是 }，不能是[ 或是 ]
{
'产品名称': "",
"产品卖点": "大家是不是觉得最近夏日炎炎，........",
    "市场价格":  "39.9",
    "直播价格": "19.9",
    "产品信息": "500ml*5瓶",
    "口味": "薏米水/陈皮水/桂圆水",
    "贈品": "买4瓶送1瓶",
    "其他优势": ""
}

＝＝＝＝＝产品卖点 好的範例 也可以模仿
＝＝产品卖点 範例1
话术（引起兴趣）：1、很多地方有坐月子喝醪糟通乳催乳的传统习惯。醪糟的温通活血功效，能加强新陈代谢，帮助血液循环，适合用于坐月子的女性恶露不下、体质虚弱、乳汁不通或较少，可以促进产妇催乳，大补气血，排出体内的淤血
2、在夏季，天气炎热，胃口不好，食用醪糟可以除湿祛暑降火，还能开胃、帮助消化。醪糟含有维生素、氨基酸、微量元素，能够促进血液循环、促进乳腺发育。
3、对于女性经期和产褥期有很好的补益作用，补养气血、暖宫散寒。对于手脚冰冷，怕冷，或者经期宫寒痛经，月经量少、有血块，经期不顺畅的女生有很好的活血化瘀温阳的作用。经期前后喝完会觉得全身上下都舒畅温暖的。
4、工作辛劳的人，干完体力活喝一碗醪糟，解渴保暖又养气血，缓解疲劳
卖点：
1、有机产品认证、国家地理标志认证，从种植到制作，坚持0添加，18道工序自然慢发酵，使用黄山山泉水小缸发酵，带你回忆小时候的味道
2、究竟不耐受人群、男女老少都可以喝，月子期、哺乳期也可以吃，孕妈妈可以少量吃，经期也可以吃，量大的要少吃
3、补养气血、温补阳气、舒筋活血、助消化、健脾养胃、强心益智
4、固形物含量大于等于60%，比行业普遍标准高42%
营销话术：
真正有机醪糟，从种植到制作，无论是使用的水、糯米还是植物酒曲，都是纯天然无科技，小时候的味道。现在下单有限时立减+满减+满赠，附赠专属中医顾问保驾护航
禁忌人群：1、室温（25℃以下）遮阴保存，冷藏更佳
2、开封后请冷藏，避免持续发酵变酸；为保证口感，若需要加热食用，建议最后一步放入醪糟，短时加热即可
3、脾胃湿热、实热、风热感冒不推荐


＝＝产品卖点 範例2
话术（引起兴趣）：
1、产后⽓⾎亏虚，体乏⽆⼒，⾯⾊苍⽩，⼼慌⽓短
2、⼥⼦⽓⾎两虚引起的⽉经不调，⽉经量少，⾊淡，不规律
3、失眠，多梦，容易健忘，身体疲劳
4、⾯⾊⽆华，⽪肤粗糙，肤⾊暗沉、蜡⻩，嘴唇没有⾎⾊
5、冬天⼿脚冰凉
6、⽓⾎亏虚引起的便秘
7、蹲下猛得站起身噶觉两眼发⿊
卖点：
1、历史悠久：玉灵膏源自清代名医王孟英的《随息居饮食谱》，经过数百年的传承和验证，具有补血、益气、安神、改善睡眠、益脾胃等多重功效，能够有效缓解因血虚、气虚引起的如面色无华、失眠多梦、心悸等
2、传统工艺制作：玉灵膏制作严格遵循古法，桂圆肉和西洋参10:1黄金配比历经72小时不断火柴火蒸制，使桂圆肉和西洋参的精华充分融合，达到更好滋补效果
3、无额外添加：玉灵膏配料纸包含西洋参和桂圆肉制成，无添加任何化学成分
禁忌人群：
1、玉灵膏推荐早上饭后1小时左右吃，半勺左右，加适量热水搅拌后饮用，一瓶可以喝大概半个月到一个月左右
2、经期量多可以停几天，量少就不用停
3、痰火、湿热体质不适合
4、孕妇不宜
5、阴虚火旺易上火人群少吃
6、晚上不吃哦，可能会兴奋睡不着


＝＝产品卖点 範例3
使用场景：
1、食欲不振，消化不良，容易腹胀、腹泻
2、⾯⾊苍⽩或萎⻩、头晕乏⼒、⼼慌⽓短、⽉经量少
3、身体虚弱的⽼⼈，有助于补充营养，改善体质，增强脾胃功能
4、病后或者产后康复者，身体虚弱，能帮助恢复元⽓
5、⻓期劳累、压⼒⼤，导致身体处于亚健康状态者：可调理脾胃，改善身体机能。
卖点：
1、融汇古今：八珍粉源自传统中医配方，历史悠久，具有深厚的文化底蕴，适用于脾胃虚弱者、气血不足者、肾虚者、免疫力低下者等多种人群。我们遵守传统配方保留了党参，是市面上少有的党参八珍粉，并且对八珍粉的配方进行调整升级，加入砂仁、鸡内金，更适合现代人群体制。
2、怀山药添加≥55%，党参添加≥10%，科学配比，相得益彰
3、粉质细腻，配料干净，无额外添加
营销话术：
市面上少有的党参八珍粉，且党参含量第二，额外添加了砂仁、鸡内金两种昂贵药材，但价格与其他竞品价格一致甚至更低，现在下单有限时立减+满减+满赠，附赠专属中医顾问保驾护航
禁忌人群：豆制品蛋白过敏者慎用，孕妇不推荐，有薏米，含量不多。（如果买了，稳定期食用，一天不超过一包）

＝＝产品卖点 範例4
使用场景：
1、失眠多梦，更年期女性。经常焦虑、犯愁，精神压力大，失眠、多梦，梦连绵不绝，尤其是更年期的女性，经常叹气、生气，用解郁汤泡脚，胜过吃安眠药助眠。
肝气不舒导致的不孕不育、月经不调，心理压力过大，肝气严重不舒，用解郁汤，疏通身体各个部位的堵塞。
2、一吃补药就上火    身体虚损，面色差，疲劳，心悸，秋冬天手脚冰凉，月经前头晕头痛，月经量少，但是一吃补药就会上火，牙龈红肿，咽喉肿痛，脸上长痘，用解郁汤将身体经络气机疏通打开，各种补品才能补进去。
3、这个方子为什么要用来泡脚，而不是直接喝呢？喝进去效果不是更好吗？
    之所以选择泡脚的方式，是出于两方面的考虑：第一，我们现在每天吃得都很油腻，还特别爱吃冰激凌等冷饮，脾胃很可能已受伤；第二，肝气不舒本身就会让脾胃遭受重创。脾胃的通道被堵塞了，很难把药物吸收进去。可以改走体外，通过泡脚的方式，从经络把药物送进去。
卖点：
1、好汤用好料，好料泡好脚，19味草本，传承经典古方，科学配比更懂女生
2、草本东方，自然纯粹，坚持0添加
3、6小时文火熬煮，层层过滤残渣，3秒速溶无渣子
禁忌人群：
1、饭前或饭后30分钟以内及酒醉后不宜使用
2、皮肤有伤口、溃烂、红肿,高血压及心脑血管者不宜使用
3、儿童、孕妇、产妇及女性生理期不宜使用
4、对中草药过敏者不宜使用
5、本品不能代替药物,仅适用于保健泡浴,仅可外用
6、本品非药品;采用中药材科学配伍,经文火360分钟(6小时) 炼制而成


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
