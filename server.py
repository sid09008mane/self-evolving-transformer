from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import threading
import psutil
import torch

from execution.trainer import Trainer
from nas.architecture_generator import ArchitectureGenerator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trainer = Trainer()
generator = ArchitectureGenerator()

training_running = False
logs = []


# ----------------------------
# SYSTEM METRICS
# ----------------------------
@app.get("/metrics")
def get_metrics():

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "gpu": gpu
    }


# ----------------------------
# START TRAINING
# ----------------------------
@app.post("/start-training")
def start_training():

    global training_running

    if training_running:
        return {"status": "already running"}

    training_running = True

    def train():

        global training_running

        population = generator.initialize_population()

        for genome in population:

            log = f"Training genome: {genome}"
            logs.append(log)

            fitness = trainer.train_genome(genome)

            genome.fitness = fitness

            logs.append(f"Fitness: {fitness}")

        training_running = False

    thread = threading.Thread(target=train)
    thread.start()

    return {"status": "training started"}


# ----------------------------
# TRAINING STATUS
# ----------------------------
@app.get("/status")
def status():
    return {"training": training_running}


# ----------------------------
# LIVE LOG STREAM
# ----------------------------
@app.websocket("/logs")
async def websocket_logs(ws: WebSocket):

    await ws.accept()

    last = 0

    while True:

        if len(logs) > last:
            await ws.send_text(logs[last])
            last += 1
