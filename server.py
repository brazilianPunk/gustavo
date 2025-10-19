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
# CONFIGURAÇÃO GERAL
# ================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPABASE_URL = "https://udpbhizhyaesheidalho.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...."  # substitua pela sua
SUPABASE_BUCKET = "Rostos"
REGISTERED_PATH = "capture (3).jpg"
REGISTERED_NAME = "Gustavo"

SIMILARITY_THRESHOLD = 0.65

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================================
# FUNÇÕES AUXILIARES
# ================================
def get_public_url(bucket, path):

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


def load_registered_face():
    """Carrega o rosto cadastrado"""
    global registered_face, registered_name
    try:
        public_url = get_public_url(SUPABASE_BUCKET, REGISTERED_PATH)
        if not public_url:
            logging.error("❌ Falha ao obter URL pública.")
            return False

        img = download_image_bgr(public_url)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

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

    return render_template("index.html", status={
        "registered_name": registered_name,
        "similarity_threshold": SIMILARITY_THRESHOLD
    })


@app.route("/upload_frame", methods=["POST"])
def upload_frame():

    global latest_frame, latest_msg

    try:
        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "erro", "mensagem": "Frame inválido"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        msg = "Nenhum rosto detectado"
        color = (0, 0, 255)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (registered_face.shape[1], registered_face.shape[0]))

            # comparação de histograma
            hist_ref = cv2.calcHist([registered_face], [0], None, [256], [0, 256])
            hist_roi = cv2.calcHist([roi_resized], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist_ref, hist_roi, cv2.HISTCMP_CORREL)

            if similarity >= SIMILARITY_THRESHOLD:
                msg = f"Acesso Permitido ({registered_name})"
                color = (0, 255, 0)
                logging.info("✅ Rosto reconhecido: similaridade %.2f", similarity)
            else:
                msg = f"Desconhecido ({similarity:.2f})"
                color = (0, 0, 255)

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
    global latest_frame, latest_msg
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
