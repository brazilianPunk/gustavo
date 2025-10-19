import os
import time
import logging
import threading
import cv2
import numpy as np
import requests
from face_recognition_lite import face_recognition
from flask import Flask, render_template, Response, request, jsonify
from supabase import create_client

# ================================
#   CONFIGURAÇÕES
# ================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPABASE_URL = "https://udpbhizhyaesheidalho.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...."  # substitua pela sua
SUPABASE_BUCKET = "Rostos"
REGISTERED_PATH = "capture (3).jpg"
REGISTERED_NAME = "Gustavo"
SIMILARITY_THRESHOLD = 0.45  # entre 0 e 1 — menor = mais sensível

app = Flask(__name__)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ================================
#   VARIÁVEIS GLOBAIS
# ================================
lock = threading.Lock()
registered_encoding = None
registered_name = None
latest_frame = None
latest_msg = "Aguardando imagem..."

# ================================
#   FUNÇÕES AUXILIARES
# ================================
def get_public_url(bucket, path):
    """Gera URL pública do rosto cadastrado"""
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
    """Carrega e codifica o rosto cadastrado"""
    global registered_encoding, registered_name
    try:
        public_url = get_public_url(SUPABASE_BUCKET, REGISTERED_PATH)
        if not public_url:
            logging.error("❌ Não foi possível obter URL pública do rosto cadastrado.")
            return False

        img = download_image_bgr(public_url)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)

        if not encodings:
            logging.error("❌ Nenhum rosto detectado na imagem cadastrada.")
            return False

        registered_encoding = encodings[0]
        registered_name = REGISTERED_NAME
        logging.info("✅ Rosto cadastrado com sucesso: %s", registered_name)
        return True

    except Exception as e:
        logging.error("Erro ao carregar rosto cadastrado: %s", e)
        return False


load_registered_face()

# ================================
#   ROTAS FLASK
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
    """Recebe frames da ESP32-CAM"""
    global latest_frame, latest_msg

    try:
        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("❌ Frame inválido recebido.")
            return jsonify({"status": "erro", "mensagem": "frame inválido"}), 400

        logging.info("📸 Frame recebido da ESP32-CAM.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        msg = "Nenhum rosto detectado"
        color = (0, 0, 255)

        for encoding in encodings:
            distance = face_recognition.face_distance([registered_encoding], encoding)[0]
            if distance <= SIMILARITY_THRESHOLD:
                msg = f"Acesso Permitido ({registered_name})"
                color = (0, 255, 0)
                logging.info("✅ %s - distância %.2f", msg, distance)
            else:
                msg = f"Desconhecido (distância {distance:.2f})"
                logging.warning("🚨 %s", msg)

        # Salva o último frame para visualização
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
    """Exibe o último frame recebido"""
    global latest_frame, latest_msg
    if latest_frame is None:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        return Response(buffer.tobytes(), mimetype="image/jpeg")

    frame_copy = latest_frame.copy()
    cv2.putText(frame_copy, latest_msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    _, buffer = cv2.imencode(".jpg", frame_copy)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


# ================================
#   EXECUÇÃO
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
