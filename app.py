from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
import re
import json

app = Flask(__name__)
CORS(app)

# 從環境變數讀取 API Key（部署時在 Render 設定）
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
if not NVIDIA_API_KEY:
    print("⚠️ 警告：未設定 NVIDIA_API_KEY 環境變數")

# 首頁：顯示前端 HTML
@app.route('/')
def index():
    return render_template('index.html')

# AI 批改 API
@app.route('/api/ai-grade', methods=['POST'])
def ai_grade():
    if not NVIDIA_API_KEY:
        return jsonify({'error': '伺服器未設定 API Key'}), 500

    data = request.get_json()
    student_answers = data.get('student_answers', [])
    teacher_answers = data.get('teacher_answers', [])

    if not student_answers or not teacher_answers:
        return jsonify({'error': '缺少答案'}), 400

    # 組合提示詞
    prompt = "你是一個嚴格且專業的老師，請逐題判斷學生的答案是否正確（考量語意是否正確，而非逐字完全相同），並給予簡短的評語（每個答案一句話）。\n\n"
    for i, (s, t) in enumerate(zip(student_answers, teacher_answers), 1):
        prompt += f"第{i}題 標準答案：{t}，學生答案：{s}\n"
    prompt += "\n請以 JSON 格式回覆，格式為：{\"results\": [{\"correct\": true/false, \"feedback\": \"評語\"}]}"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.1-405b-instruct",  # 改用更適合改作業的 Llama 3.1
        "messages": [
            {"role": "system", "content": "你是一個嚴格的老師，請根據標準答案判斷學生答案是否正確，並給出評語。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }

    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        ai_message = result['choices'][0]['message']['content']
        
        # 解析 JSON
        json_match = re.search(r'\{.*\}', ai_message, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return jsonify(parsed)
        else:
            return jsonify({'raw': ai_message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render 會使用 PORT 環境變數
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)