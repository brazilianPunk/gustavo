import os
import time
import logging
import threading
import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response, request, jsonify
from supabase import create_client

# ================================
# CONFIGURAÇÃO
# ================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPABASE_URL = "https://udpbhizhyaesheidalho.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...."
SUPABASE_BUCKET = "Rostos"
REGISTERED_PATH = "capture (3).jpg"
REGISTERED_NAME = "Gustavo"

SIMILARITY_THRESHOLD = 0.65  # 0.65 é ideal para histogramas

app = Flask(__name__)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ================================
# VARIÁVEIS GLOBAIS
# ================================
lock = threading.Lock()
registered_face = None
registered_name = None
latest_frame = None
latest_msg = "Aguardando imagem..."

# Carrega o classificador de rostos do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================================
# FUNÇÕES AUXILIARES
# ================================
def get_public_url(bucket, path):
    """Obtém a URL pública do rosto cadastrado"""
    try:
        res = supabase.storage.from_(bucket).get_public_url(path)
        if isinstance(res, dict):
            return res.get("publicUrl") or res.get("public_url")
        return res
    except Exception as e:
        logging.error("Erro ao gerar public URL: %s", e)
        return None


def download_image_bgr(url):
    """Baixa imagem do Supabase"""
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def preprocess_face(img):
    """Converte para escala de cinza e equaliza o histograma"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def load_registered_face():
    """Carrega e processa o rosto cadastrado do Supabase"""
    global registered_face, registered_name
    try:
        public_url = get_public_url(SUPABASE_BUCKET, REGISTERED_PATH)
        if not public_url:
            logging.error("❌ Falha ao obter URL pública.")
            return False

        img = download_image_bgr(public_url)
        gray = preprocess_face(img)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        if len(faces) == 0:
            logging.error("❌ Nenhum rosto encontrado na imagem cadastrada.")
            return False

        (x, y, w, h) = faces[0]
        registered_face = gray[y:y+h, x:x+w]
        registered_name = REGISTERED_NAME
        logging.info("✅ Rosto cadastrado: %s", registered_name)
        return True
    except Exception as e:
        logging.error("Erro ao carregar rosto cadastrado: %s", e)
        return False


load_registered_face()

# ================================
# ROTAS FLASK
# ================================
@app.route("/")
def index():
    """Página principal"""
    return render_template("index.html", status={
        "registered_name": registered_name,
        "similarity_threshold": SIMILARITY_THRESHOLD
    })


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """Recebe frames da ESP32-CAM e realiza o reconhecimento"""
    global latest_frame, latest_msg

    try:
        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "erro", "mensagem": "Frame inválido"}), 400

        gray = preprocess_face(frame)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        msg = "Nenhum rosto detectado"
        color = (0, 0, 255)

        if registered_face is None:
            logging.warning("⚠️ Nenhum rosto cadastrado carregado.")
            return jsonify({"status": "erro", "mensagem": "Rosto cadastrado não encontrado."}), 500

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (registered_face.shape[1], registered_face.shape[0]))

            # comparação por histograma
            hist_ref = cv2.calcHist([registered_face], [0], None, [256], [0, 256])
            hist_roi = cv2.calcHist([roi_resized], [0], None, [256], [0, 256])
            hist_ref = cv2.normalize(hist_ref, hist_ref)
            hist_roi = cv2.normalize(hist_roi, hist_roi)

            similarity = cv2.compareHist(hist_ref, hist_roi, cv2.HISTCMP_CORREL)

            if similarity >= SIMILARITY_THRESHOLD:
                msg = f"Acesso Permitido ({registered_name})"
                color = (0, 255, 0)
                logging.info("✅ Reconhecido: %.2f", similarity)
            else:
                msg = f"Desconhecido ({similarity:.2f})"
                color = (0, 0, 255)
                logging.warning("🚨 Desconhecido: %.2f", similarity)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, msg, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with lock:
            latest_frame = frame.copy()
            latest_msg = msg

        time.sleep(0.3)
        return jsonify({"status": "ok", "mensagem": msg}), 200

    except Exception as e:
        logging.error(f"Erro no upload_frame: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500


@app.route("/latest_frame")
def latest_frame_view():
    """Retorna o último frame recebido (com caixas e texto)"""
    global latest_frame
    if latest_frame is None:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        return Response(buffer.tobytes(), mimetype="image/jpeg")

    _, buffer = cv2.imencode(".jpg", latest_frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


# ================================
# EXECUÇÃO
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
