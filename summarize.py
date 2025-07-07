from pkg.plugin.context import register, handler, llm_func, BasePlugin, APIHost, EventContext
from pkg.plugin.events import *  # 导入事件类
import requests
import pdfplumber
import json
import re
import os
from trafilatura import fetch_url, extract

API_KEY = ''
URL = ''


custom_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache", 
        }

def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取所有文本。
    """
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        print(f"成功从 '{pdf_path}' 提取文本。")
        return text_content
    except Exception as e:
        print(f"从 PDF 提取文本时发生错误: {e}")
        return None
    

def fetch_data(content_type = 'normal', pdf_data=None, temperature: float = 0.7, max_tokens: int = 10000):
    url = URL
    messages_arxiv = f'''
        请以纯文本格式总结以下深度学习领域的论文。总结应清晰、简洁、信息量大，并严格按照以下结构和具体要求撰写：
        ** 论文： **
        {pdf_data}

        一、论文基本信息



        论文标题：[请填写论文的完整标题，精确无误]

        作者：[请列出所有作者，或若作者众多，则列出第一作者、通讯作者及其他关键贡献者，并指出其单位]

        发表信息：[请具体注明发表年份、会议名称（例如：NeurIPS 2024, ICML 2023, AAAI 2025）或期刊名称、卷号、期号和页码范围（例如：Nature Machine Intelligence, Vol. 6, No. 1, pp. 1-10），确保信息的完整性]

        二、核心内容总结



        研究背景、动机与问题定义

        核心问题：论文试图解决的具体深度学习问题是什么？该问题在现有研究或实际应用中面临哪些核心挑战或瓶颈？（例如：长序列依赖问题、数据稀疏性、模型泛化能力不足、训练效率低下、对抗鲁棒性差等）

        研究动机：为什么解决这个问题很重要？现有主流方法（SOTA或经典方法）在处理该问题时存在哪些根本性缺陷（例如：计算开销大、理论解释性差、性能上限低、对特定数据分布敏感等）？

        上下文：将该问题置于当前深度学习研究的哪个具体子领域（例如：自监督学习、多模态融合、图神经网络、高效Transformer、因果推断等）中进行讨论？

        （本部分建议控制在 80-120 字）

        提出的方法/模型详解

        核心思想与创新点：论文提出的新方法或模型（请给出具体名称，如有）的核心机制或创新理念是什么？它如何从根本上区别于现有方法？（例如：引入了新的注意力机制、设计了独特的网络层、提出了新的正则化策略、优化了损失函数形式、构建了全新的预训练范式等）

        关键组件与工作原理：详细描述该方法或模型的主要构成模块（例如：编码器、解码器、特征提取器、门控单元等）及其具体功能与相互关系。解释其数据流或信息处理流程。如果涉及到数学概念，请用简洁的文字解释其直观含义和作用，无需列出公式。

        理论或技术支撑：该方法的提出是否基于特定的理论洞察、新的数学推导、或结合了哪些已有的技术？

        （本部分建议控制在 180-250 字）

        实验设置、结果分析与讨论

        实验设计：论文在哪些具体任务（例如：图像分类、目标检测、自然语言理解、时间序列预测等）和数据集上进行了实验？每个任务使用了哪些主要数据集（请给出数据集名称，例如：ImageNet, GLUE, COCO等）？

        评估指标：主要使用了哪些量化评估指标（例如：Accuracy, F1-score, IoU, BLEU, FID等）来衡量模型性能？

        对比方法：与哪些强基线（strong baselines）或SOTA方法进行了比较？请简要说明比较的维度。

        核心发现与数据支持：最关键的实验结果是什么？（例如：在哪个任务或数据集上达到了怎样的SOTA性能？相对现有方法提升了多少百分点？在特定条件下的鲁棒性如何？）。是否有消融实验（Ablation Study）或可视化分析来验证所提出组件的有效性或模型的解释性？主要结论是什么？

        （本部分建议控制在 150-200 字）

        论文的主要贡献与影响

        学术贡献：该论文在理论、方法或应用上对深度学习领域做出了哪些实质性贡献？是否填补了研究空白，或为解决某个问题提供了新的视角/范式？

        潜在影响：该研究成果未来可能对相关领域或实际应用产生怎样的深远影响？是否为后续研究指明了方向或奠定了基础？

        （本部分建议控制在 60-90 字）

        局限性与未来工作

        方法局限：论文提出的方法在哪些方面存在不足或限制？（例如：对特定数据分布敏感、计算资源需求高、缺乏理论保障、泛化能力有限、可解释性仍待提高等）

        未来研究方向：作者提出了哪些具体的未来研究方向来弥补现有局限或进一步扩展研究？（例如：探索更高效的训练策略、将其应用于其他模态、提升模型的可信赖性等）

        （本部分建议控制在 40-70 字）

        三、总结要求



        格式：纯文本输出。不包含任何Markdown以外的格式，例如LaTeX代码、HTML标签等。不包含表格、图表、公式、代码块或其他非纯文本元素。

        语言：使用清晰、准确、简洁的中文学术语言。避免口语化、含糊不清的表述。所有术语应保持一致。

        内容聚焦：严格聚焦于论文本身的核心思想、方法、实验结果和贡献。避免加入个人评论或引申。

        字数控制：总字数控制在 750-1000 字之间。请严格遵循各部分建议的字数范围，以确保内容的平衡性。

        可读性：逻辑严谨，分点阐述，段落间过渡自然。每一点都应有具体内容支撑，而非泛泛而谈，确保信息密度高且易于理解。

        避免：

        过度引用原文：用自己的话提炼和概括，而非直接复制粘贴。

        个人主观评价：所有论述需基于论文内容。

        冗余信息：去除一切不必要的重复或修饰性语言。

    '''
    messages_normal = messages_normal = f'''
    请根据以下文件的内容生成结构化摘要：

    输入内容：
    {pdf_data}

    摘要要求：

    核心主题（20-40字）

    用中性客观的语言概括材料的核心议题

    避免使用比喻或夸张表述

    关键要素（分条目列出）

    [必需] 提取3-5个具有数据支撑的核心事实

    [可选] 标注重要人物/机构及其关联观点

    每个条目保持15-30字的紧凑表述

    技术细节（如存在）

    专业术语需附加10字内的白话解释

    方法论的实现步骤用"动词+宾语"短句表述

    价值判断

    仅当原文明确给出时才保留结论

    区分作者观点与客观事实

    格式规范：
    • 禁用任何符号标记（包括*#等）
    • 数字统一用阿拉伯数字（如"3个步骤"）
    • 英文专有名词首字母大写（如"Transformer模型"）
    • 避免出现"本文""笔者"等主观指代

    质量控制：

    事实性错误需标注[需核实]

    存在争议的观点标注[存在分歧]

    模糊表述需改写为具体陈述

    请严格按以下结构输出：
    [主题]
    ...
    [要素]

    ...

    ...
    ...
    [补充说明]（可选）
    ...

    **总结要求**
    格式：纯文本输出。不包含任何Markdown以外的格式，例如LaTeX代码、HTML标签等。不包含表格、图表、公式、代码块或其他非纯文本元素。

    语言：使用清晰、准确、简洁的中文语言。避免口语化、含糊不清的表述。

    字数控制：总字数控制在 750-1000 字之间。请严格遵循各部分建议的字数范围，以确保内容的平衡性。

    可读性：逻辑严谨，分点阐述，段落间过渡自然。每一点都应有具体内容支撑，而非泛泛而谈，确保信息密度高且易于理解。

    '''
    if content_type == 'arxiv':
        messages = messages_arxiv
    else:
        messages = messages_normal
    messages_payload = [
        {"role": "system", "content": messages},
        {"role": "user", "content": f"开始总结"}
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": 'gemini-2.5-flash',
        "messages": messages_payload,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = None
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求 API 时发生错误: {e}")
        if response is not None:
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        return response.text


def process_arxiv_link(url: str, output_dir: str = "downloaded_arxiv", ):
    """
    判断链接是否为 arXiv 链接，如果是则下载、保存，并发送给另一个 LLM 处理。
    如果不是或处理失败，则报错。

    Args:
        url (str): 待处理的链接。
        output_dir (str): 下载文件保存的目录。
        llm_api_endpoint (str): 目标 LLM 的 API 端点。
    """
    # 1. 判断是否为 arXiv 链接
    # 常见的 arXiv 链接模式：https://arxiv.org/abs/XXXX.XXXXX 或 https://arxiv.org/pdf/XXXX.XXXXX.pdf
    arxiv_pattern = re.compile(r"https?://arxiv\.org/(abs|pdf)/(\d{4}\.\d{5}(v\d+)?)\.?(pdf)?")
    match = arxiv_pattern.match(url)

    if not match:
        return '<<NOT A URL LINK>>'

    print(f"检测到有效的 arXiv 链接: {url}")

    # 确保是 PDF 下载链接，如果不是，尝试转换为 PDF 链接
    paper_id = match.group(2) # 提取论文ID
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    print(f"尝试下载 PDF 地址: {pdf_url}")

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{paper_id}.pdf"
    file_path = os.path.join(output_dir, file_name)

    try:
        # 3. 下载文件
        print(f"正在下载文件: {pdf_url}...")
        response = requests.get(pdf_url, stream=True, timeout=15, headers=custom_headers)
        response.raise_for_status() # 检查 HTTP 请求是否成功

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件下载成功并保存至: {file_path}")

        # 4. 读取文件并发送 POST 请求给 LLM
        print(f"正在将文件发送至 LLM ...")
        file_path = output_dir + '/' + file_name
        pdf_info = extract_text_from_pdf(file_path)
        data = fetch_data(content_type='arxiv', pdf_data=pdf_info)
        print("文件成功发送给 LLM。LLM 的响应：")
        return data

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络请求错误或下载失败：{e}")
    except IOError as e:
        raise RuntimeError(f"文件保存或读取错误：{e}")
    except Exception as e:
        raise RuntimeError(f"处理过程中发生未知错误：{e}")
    
def process_normal_link(url:str, output_dir:str = 'downloaded_url'):
    os.makedirs(output_dir, exist_ok=True)
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    is_url = re.match(pattern, url)
    if not is_url:
        return '<<ERRORR>>'
    else:
        pdf_patten = re.compile(r'\b(https?://([^/]+)(/[^\s\'"]+?\.pdf)(?:\?[^\s\'"]*)?\b)')
        match = pdf_patten.match(url)
        if not match:
            # try:
            html = fetch_url(url)  # 或直接输入HTML文本
            markdown = extract(html, output_format="markdown")
            safe_name = re.sub(r'[^\w\-_.]', '_', url.replace('://', '_'))  # 替换非法字符
            file_name = f"{safe_name}.md"
            file_path = os.path.join(output_dir, file_name) 
            with open(file_path, 'w+', encoding='utf-8') as f:
                f.write(markdown)
            print(f"文件下载成功并保存至: {file_path}")
            print(f"正在将文件发送至 LLM ...")
            data = fetch_data(content_type='normal', pdf_data=markdown)
            print("文件成功发送给 LLM。LLM 的响应：")
            return data
            # except:
            #     raise RuntimeError(f"url请求下载错误")
        else:
            try:
                response = requests.get(url, stream=True, timeout=15, headers=custom_headers)
                response.raise_for_status()
                url_name = match.group(2)
                file_name = f"{url_name}.pdf"
                file_path = os.path.join(output_dir, file_name) 
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)            
                print(f"文件下载成功并保存至: {file_path}")
                print(f"正在将文件发送至 LLM ...")
                file_path = output_dir + '/' + file_name
                pdf_info = extract_text_from_pdf(file_path)
                data = fetch_data(content_type='arxiv', pdf_data=pdf_info)
                print("文件成功发送给 LLM。LLM 的响应：")
                return data
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"网络请求错误或下载失败：{e}")
            except IOError as e:
                raise RuntimeError(f"文件保存或读取错误：{e}")
            except Exception as e:
                raise RuntimeError(f"处理过程中发生未知错误：{e}")

    
@register(name="paper_sm", description="论文summarize", version="0.1", author="Regenin")
class Paper_summarize(BasePlugin):

    # 插件加载时触发
    def __init__(self, host: APIHost):
        pass

    # 异步初始化
    async def initialize(self):
        pass

    # 当收到个人消息时触发
    @handler(PersonNormalMessageReceived)
    async def person_normal_message_received(self, ctx: EventContext):
        msg = ctx.event.text_message  # 这里的 event 即为 PersonNormalMessageReceived 的对象
        if msg[:10] == "/summarize":  # 
            data = process_arxiv_link(msg[11:])
            if data != '<<NOT A URL LINK>>':
                extracted_text = data.get('choices', [{}])[0].get('message', {}).get('content')
                ctx.add_return("reply", ["{}".format(extracted_text)])
            else:
                data = process_normal_link(msg[11:])
                if data == '<<ERRORR>>':
                    ctx.add_return("reply", ['错误，未找到url链接'])
                else:
                    extracted_text = data.get('choices', [{}])[0].get('message', {}).get('content')
                    ctx.add_return("reply", ["{}".format(extracted_text)])
            self.ap.logger.debug("hello, {}".format(ctx.event.sender_id))
            ctx.prevent_default()



    # 当收到群消息时触发
    @handler(GroupNormalMessageReceived)
    async def group_normal_message_received(self, ctx: EventContext):
        msg = ctx.event.text_message  # 这里的 event 即为 GroupNormalMessageReceived 的对象
        if msg[:10] == "/summarize":  
            data = process_arxiv_link(msg[11:])
            if data != '<<NOT A URL LINK>>':
                await ctx.send_message('group', ctx.event.launcher_id, ['找到链接，总结生成中，可能需要几分钟'])
                extracted_text = data.get('choices', [{}])[0].get('message', {}).get('content')
                ctx.add_return("reply", ["{}".format(extracted_text)])
            else:
                data = process_normal_link(msg[11:])
                if data == '<<ERRORR>>':
                    ctx.add_return("reply", ['错误，未找到url链接'])
                else:
                    await ctx.send_message('group', ctx.event.launcher_id, ['找到链接，总结生成中，可能需要几分钟'])
                    extracted_text = data.get('choices', [{}])[0].get('message', {}).get('content')
                    ctx.add_return("reply", ["{}".format(extracted_text)])
            self.ap.logger.debug("hello, {}".format(ctx.event.sender_id))
            ctx.prevent_default()

    # 插件卸载时触发
    def __del__(self):
        pass