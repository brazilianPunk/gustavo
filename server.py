import os
import time
import logging
import threading
import cv2
import numpy as np
import requests
import face_recognition
from flask import Flask, render_template, Response, request, jsonify

# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPABASE_URL = "https://udpbhizhyaesheidalho.supabase.co"
SUPABASE_BUCKET_PATH = "storage/v1/object/public/Rostos/capture%20(3).jpg"
REGISTERED_NAME = "Gustavo"
SIMILARITY_THRESHOLD = 0.45  # menor = mais sensível

app = Flask(__name__)

# -------------------------------
# VARIÁVEIS GLOBAIS
# -------------------------------
lock = threading.Lock()
registered_encoding = None
latest_frame = None
latest_msg = "Aguardando imagem..."

# -------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------
def download_image_bgr(url):
    """Baixa imagem do Supabase diretamente pelo link público"""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        logging.info("✅ Imagem cadastrada baixada com sucesso.")
        return img
    except Exception as e:
        logging.error(f"❌ Erro ao baixar imagem: {e}")
        return None


def load_registered_face():
    """Carrega e codifica o rosto cadastrado"""
    global registered_encoding
    img = download_image_bgr(f"{SUPABASE_URL}/{SUPABASE_BUCKET_PATH}")
    if img is None:
        logging.error("❌ Falha ao carregar imagem cadastrada.")
        return False

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    if not encodings:
        logging.error("❌ Nenhum rosto detectado na imagem cadastrada.")
        return False

    registered_encoding = encodings[0]
    logging.info("✅ Rosto cadastrado com sucesso.")
    return True


load_registered_face()

# -------------------------------
# ROTAS FLASK
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", status={
        "registered_name": REGISTERED_NAME,
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
            return jsonify({"status": "erro", "mensagem": "Frame inválido"}), 400

        logging.info("📸 Frame recebido da ESP32-CAM")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        msg = "Nenhum rosto detectado"
        color = (0, 0, 255)

        for encoding, (top, right, bottom, left) in zip(encodings, faces):
            distance = face_recognition.face_distance([registered_encoding], encoding)[0]

            if distance <= SIMILARITY_THRESHOLD:
                msg = f"Acesso Permitido ({REGISTERED_NAME})"
                color = (0, 255, 0)
                logging.info(f"✅ {msg} (distância {distance:.2f})")
            else:
                msg = f"Desconhecido ({distance:.2f})"
                logging.warning(f"🚨 {msg}")

            # desenha o retângulo
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, msg, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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


# -------------------------------
# EXECUÇÃO
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
