from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
import re
import json
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# 從環境變數讀取 API Key
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
if not NVIDIA_API_KEY:
    print("⚠️ 警告：未設定 NVIDIA_API_KEY 環境變數")

# ================================================================
# PaddleOCR 初始化（延遲載入，節省記憶體）
# ================================================================
ocr = None

def get_ocr():
    """延遲載入 PaddleOCR，只有在需要時才載入"""
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR
        print("📦 正在載入 PaddleOCR...")
        ocr = PaddleOCR(
            use_angle_cls=True,        # 啟用方向分類
            lang='ch',                 # 中文
            show_log=False,            # 關閉日誌
            use_gpu=False,             # Render 免費方案無 GPU
            enable_mkldnn=True,        # 加速 CPU 推理
            cpu_threads=2
        )
        print("✅ PaddleOCR 載入完成")
    return ocr

# ================================================================
# 首頁
# ================================================================
@app.route('/')
def index():
    return render_template('index.html')

# ================================================================
# API: OCR 辨識（使用 PaddleOCR）
# ================================================================
@app.route('/api/ocr', methods=['POST'])
def ocr_image():
    try:
        # 接收前端傳來的圖片（Base64）
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': '缺少圖片數據'}), 400

        # 去除 Base64 前綴
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # 解碼 Base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 呼叫 PaddleOCR 辨識
        ocr_engine = get_ocr()
        result = ocr_engine.ocr(image, cls=True)

        # 解析結果，提取文字
        extracted_text = []
        if result and result[0]:
            for line in result[0]:
                extracted_text.append(line[1][0])  # line[1][0] 是辨識出的文字

        full_text = '\n'.join(extracted_text)
        return jsonify({'text': full_text})

    except Exception as e:
        print(f"OCR 錯誤: {e}")
        return jsonify({'error': str(e)}), 500

# ================================================================
# API: AI 智能批改
# ================================================================
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
        "model": "meta/llama-3.1-405b-instruct",
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)