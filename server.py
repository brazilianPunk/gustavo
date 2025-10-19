import os
import time
import logging
import threading
import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response, request, jsonify
from supabase import create_client
from deepface import DeepFace

# -----------------------
# Configuração geral
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPABASE_URL = "https://udpbhizhyaesheidalho.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...."  # substitua pela sua chave
SUPABASE_BUCKET = "Rostos"
REGISTERED_PATH = "capture (3).jpg"
REGISTERED_NAME = "Gustavo"
SIMILARITY_THRESHOLD = 0.65

app = Flask(__name__)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# -----------------------
# Variáveis globais
# -----------------------
lock = threading.Lock()
registered_embedding = None
registered_name = None
latest_frame = None
latest_msg = "Aguardando imagem..."
face_model = DeepFace.build_model("Facenet")

# -----------------------
# Funções auxiliares
# -----------------------
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
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_registered_embedding():
    global registered_embedding, registered_name
    try:
        public_url = get_public_url(SUPABASE_BUCKET, REGISTERED_PATH)
        if not public_url:
            logging.error("❌ Não foi possível obter URL pública do rosto cadastrado.")
            return False
        img = download_image_bgr(public_url)
       rep = DeepFace.represent(
    img_path=frame,
    model_name="Facenet",
    enforce_detection=False,
    detector_backend="opencv"   # evita MTCNN pesado
)[0]

        registered_embedding = np.array(rep["embedding"], dtype=np.float32)
        registered_name = REGISTERED_NAME
        logging.info("✅ Embedding carregado para: %s", registered_name)
        return True
    except Exception as e:
        logging.error("Erro ao carregar rosto cadastrado: %s", e)
        return False


load_registered_embedding()

# -----------------------
# Rota principal
# -----------------------
@app.route("/")
def index():
    return render_template("index.html", status={
        "registered_name": registered_name,
        "similarity_threshold": SIMILARITY_THRESHOLD
    })

# -----------------------
# Rota para upload da ESP32
# -----------------------
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global latest_frame, latest_msg
    try:
        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("❌ Frame inválido recebido.")
            return jsonify({"status": "erro", "mensagem": "frame inválido"}), 400

        logging.info("📸 Frame recebido da ESP32-CAM.")

        # Reconhecimento facial
        rep = DeepFace.represent(img_path=frame, model_name="Facenet", enforce_detection=False)[0]
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

        # Atualiza variáveis globais
        with lock:
            latest_frame = frame.copy()
            latest_msg = msg
        time.sleep(0.5)

        return jsonify({"status": "ok", "mensagem": msg}), 200

    except Exception as e:
        logging.error(f"Erro no upload_frame: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

# -----------------------
# Rota para exibir último frame recebido
# -----------------------
@app.route("/latest_frame")
def latest_frame_view():
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

# -----------------------
# Inicialização
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
