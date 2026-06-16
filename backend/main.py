import os
import grpc
import json
import httpx
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import vap_inference_pb2
import vap_inference_pb2_grpc

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("BACKEND_ORCHESTRATOR")

app = FastAPI()

# Config
FCP_GRPC_ADDR = os.getenv("FCP_GRPC_ADDR", "fcp-former:50051")
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://vllm:8000/v1/chat/completions")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "google/gemma-2-9b-it")

# Keep gRPC channel open
channel = grpc.aio.insecure_channel(FCP_GRPC_ADDR)
stub = vap_inference_pb2_grpc.VAPPredictorStub(channel)

class TelemetryPayload(BaseModel):
    peep: float
    pip: float
    fio2: float
    hrv: float
    procalcitonin: float
    p_f_ratio: float

async def get_explainability(payload: TelemetryPayload, risk: float):
    prompt = f"Patient has risk index {risk:.4f}. PaO2/FiO2 is {payload.p_f_ratio:.1f}, Procalcitonin is {payload.procalcitonin:.2f} ng/mL. Provide a short clinical rationale."
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a clinical diagnostic AI. Provide highly concise, deterministic clinical rationale."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(VLLM_API_URL, json=data, headers=headers, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"vLLM explainability failed: {e}")
        return "Explainability generation failed or timed out."

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via WebSocket.")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # In real scenario, this would decode Delta-Encoded data.
                # For demo, we assume JSON.
                telemetry_dict = json.loads(data)
                payload = TelemetryPayload(**telemetry_dict)
                
                # Call FCP-Former
                req = vap_inference_pb2.InferenceRequest(
                    peep=payload.peep,
                    pip=payload.pip,
                    fio2=payload.fio2,
                    hrv=payload.hrv,
                    procalcitonin=payload.procalcitonin,
                    p_f_ratio=payload.p_f_ratio
                )
                
                resp = await stub.Predict(req)
                risk_prob = resp.risk_probability
                
                alert_status = "CRITICAL" if risk_prob > 0.70 or payload.p_f_ratio < 200 else "STABLE"
                
                if alert_status == "CRITICAL":
                    rationale = await get_explainability(payload, risk_prob)
                else:
                    rationale = f"Baseline state verified within physiological tolerance (Risk index: {risk_prob:.4f})."
                
                result = {
                    "status": "success",
                    "alert_level": alert_status,
                    "probability": risk_prob,
                    "rationale": rationale
                }
                
                await websocket.send_text(json.dumps(result))
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_text(json.dumps({"status": "error", "detail": str(e)}))
    except WebSocketDisconnect:
        logger.info("Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 because this runs in a docker container behind nginx
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
