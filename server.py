import os
import time
import logging
import threading
import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace

# ================================
#   CONFIGURAÇÕES GERAIS
# ================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silencia logs do TensorFlow
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# URL direta da imagem cadastrada no Supabase
REGISTERED_PUBLIC_URL = "https://udpbhizhyaesheidalho.supabase.co/storage/v1/object/public/Rostos/capture%20(3).jpg"
REGISTERED_NAME = "Gustavo"
SIMILARITY_THRESHOLD = 0.65

app = Flask(__name__)

# ================================
#   VARIÁVEIS GLOBAIS
# ================================
lock = threading.Lock()
registered_embedding = None
registered_name = None
latest_frame = None
latest_msg = "Aguardando imagem..."

# modelo leve
face_model = DeepFace.build_model("VGG-Face")

# ================================
#   FUNÇÃO: Carrega rosto cadastrado
# ================================
def load_registered_embedding():
    """Carrega o rosto cadastrado diretamente da URL pública"""
    global registered_embedding, registered_name
    try:
        logging.info("🖼️ Baixando imagem cadastrada...")
        r = requests.get(REGISTERED_PUBLIC_URL, timeout=10)
        r.raise_for_status()

        arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        rep = DeepFace.represent(
            img_path=img,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="opencv"
        )[0]

        registered_embedding = np.array(rep["embedding"], dtype=np.float32)
        registered_name = REGISTERED_NAME
        logging.info("✅ Embedding carregado para: %s", registered_name)
        return True

    except Exception as e:
        logging.error("❌ Erro ao carregar rosto cadastrado: %s", e)
        return False

# Carrega rosto na inicialização
load_registered_embedding()

# ================================
#   ROTA: Página principal
# ================================
@app.route("/")
def index():
    return render_template("index.html", status={
        "registered_name": registered_name,
        "similarity_threshold": SIMILARITY_THRESHOLD
    })

# ================================
#   ROTA: Upload de frame da ESP32
# ================================
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global latest_frame, latest_msg

    try:
        # converte bytes para imagem
        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("❌ Frame inválido recebido.")
            return jsonify({"status": "erro", "mensagem": "frame inválido"}), 400

        logging.info("📸 Frame recebido da ESP32-CAM.")

        # Reconhecimento facial
        rep = DeepFace.represent(
            img_path=frame,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="opencv"
        )[0]

        emb = np.array(rep["embedding"], dtype=np.float32)
        sim = np.dot(registered_embedding, emb) / (
            np.linalg.norm(registered_embedding) * np.linalg.norm(emb) + 1e-10
        )

        if sim >= SIMILARITY_THRESHOLD:
            msg = f"Acesso Permitido ({registered_name}) {sim:.2f}"
            color = (0, 255, 0)
            logging.info("✅ " + msg)
        else:
            msg = f"Desconhecido ({sim:.2f})"
            color = (0, 0, 255)
            logging.warning("🚨 " + msg)

        # Desenha texto sobre o frame
        cv2.putText(frame, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Atualiza frame global
        with lock:
            latest_frame = frame.copy()
            latest_msg = msg

        # pausa curta para reduzir carga
        time.sleep(0.5)
        return jsonify({"status": "ok", "mensagem": msg}), 200

    except Exception as e:
        logging.error(f"Erro no upload_frame: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

# ================================
#   ROTA: Exibir último frame
# ================================
@app.route("/latest_frame")
def latest_frame_view():
    global latest_frame, latest_msg
    if latest_frame is None:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        return Response(buffer.tobytes(), mimetype="image/jpeg")

    frame_copy = latest_frame.copy()
    _, buffer = cv2.imencode(".jpg", frame_copy)
    return Response(buffer.tobytes(), mimetype="image/jpeg")

# ================================
#   EXECUÇÃO DO SERVIDOR
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
