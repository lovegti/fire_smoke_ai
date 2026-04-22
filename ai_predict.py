import requests

API_KEY = ""
URL = ""

def predict_fire_risk(history_list):
    """传入历史温度/烟雾/CO，AI预测未来火灾概率 0~100"""
    prompt = f"""
你是消防安全分析专家。
根据以下连续环境监测时序数据，预测未来短期内发生火灾的概率，只输出0-100的纯数字：
{history_list}
要求：只返回一个整数，不要文字、不要解释。
"""

    payload = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role":"user","content":prompt}]},
        "parameters": {"result_format":"text"}
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(URL, json=payload, headers=headers, timeout=10)
        data = res.json()
        text = data["output"]["text"].strip()
        num = int(''.join(filter(str.isdigit, text)))
        return max(0, min(100, num))
    except:
        return 15