import os
import glob
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# DeepSeek 客户端：用环境变量里的 key
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 优先用单独的 AI webhook，没有就退回用原来的企业微信 webhook
WEWORK_AI_WEBHOOK = (
    os.getenv("WEWORK_WEBHOOK_URL_AI")
    or os.getenv("WEWORK_WEBHOOK_URL")
    or ""
)

OUTPUT_ROOT = "output"


def find_latest_date_dir():
    """在 output 下找到最新的日期目录，例如 output/2025年11月21日"""
    if not os.path.exists(OUTPUT_ROOT):
        print("output 目录不存在")
        return None

    candidates = []
    for name in os.listdir(OUTPUT_ROOT):
        path = os.path.join(OUTPUT_ROOT, name)
        if os.path.isdir(path):
            candidates.append(path)

    if not candidates:
        print("output 下没有日期子目录")
        return None

    # 按修改时间排序，取最新一个
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def find_report_file():
    """
    在最新日期目录下的 html 子目录中找到要给 AI 看的报告文件：
    1. 优先包含“汇总”的 html（当日汇总 / 当前榜单汇总）
    2. 没有的话，取最新修改的 html
    """
    date_dir = find_latest_date_dir()
    if not date_dir:
        return None

    html_dir = os.path.join(date_dir, "html")
    if not os.path.exists(html_dir):
        print(f"{html_dir} 目录不存在")
        return None

    html_files = glob.glob(os.path.join(html_dir, "*.html"))
    if not html_files:
        print(f"{html_dir} 下没有 html 报告文件")
        return None

    # 优先选文件名包含“汇总”的
    summary_files = [
        f for f in html_files
        if ("汇总" in os.path.basename(f))
    ]
    target_list = summary_files or html_files

    # 在候选里按修改时间排序，取最新一个
    target_list.sort(key=os.path.getmtime, reverse=True)
    report_file = target_list[0]
    print("选中的报告文件：", report_file)
    return report_file


def extract_text_from_html(path: str) -> str:
    """把 HTML 报告里的文字提出来，给大模型看"""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def call_deepseek(news_text: str) -> str:
    """调用 DeepSeek 做新闻总结 + 解读"""

    # 为了安全/省钱，最多取前 15000 字符
    clipped = news_text[:15000]

    prompt = f"""
你是一个专门给“妈妈级别普通投资者”看的【股票市场解读助手】。

下面是一份今天从各大平台抓取的热点新闻（含 A 股、美股、港股等），内容比较长：

---------------- 原始报告开始 ----------------
{clipped}
---------------- 原始报告结束 ----------------

请用【中文】输出一份简洁的解读，不要贴原文，结构参考：

1. 今天大盘和主要海外市场的整体情况（1~3 句话，通俗一点）
2. 和她持仓相关的板块 & 个股（电池/锂矿/能源金属/半导体/机器人/基建/信托/医药），按板块分点说明：
   - 每个板块 1~2 句话：今天发生了什么、是偏利好还是偏利空
3. 需要特别留意的风险点（用 🔺 列出 2~5 条）
4. 今天可以关注的机会/观察点（用 ✅ 列出 2~5 条，不要给买卖建议，只说“可以多留意、持续观察”）

要求：
- 面向的是我妈妈，尽量避免特别专业的术语，多用“看涨/看跌”“资金在流入/流出”“情绪偏悲观/乐观”这种说法
- 不要给出具体买卖建议和目标价，只做信息解读和风险提示
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个稳重的中文财经解读助手。"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    return resp.choices[0].message.content


def send_to_wework(markdown_text: str):
    """把 AI 的总结发到企业微信机器人"""
    if not WEWORK_AI_WEBHOOK:
        print("没有配置企业微信 webhook，跳过发送")
        return

    payload = {
        "msgtype": "markdown",
        "markdown": {"content": markdown_text},
    }
    try:
        resp = requests.post(WEWORK_AI_WEBHOOK, json=payload, timeout=10)
        print("WeWork response:", resp.text)
    except Exception as e:
        print("发送企业微信失败：", e)


def main():
    report_file = find_report_file()
    if not report_file:
        print("没有找到可用的报告文件，跳过 AI 分析")
        return

    text = extract_text_from_html(report_file)
    if not text:
        print("报告内容为空，跳过 AI 分析")
        return

    print("正在调用 DeepSeek 进行 AI 解读……")
    analysis = call_deepseek(text)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = f"**AI 热点解读（{now}）**\n\n{analysis}"

    send_to_wework(md)


if __name__ == "__main__":
    main()
