import os
import ssl
import json
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import websockets

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("EDGE_AGGREGATOR")

app = FastAPI(title="Edge Aggregator")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Path to client certs
CERT_PATH = os.getenv("CERT_PATH", "../certs/client.crt")
KEY_PATH = os.getenv("KEY_PATH", "../certs/client.key")
CA_PATH = os.getenv("CA_PATH", "../certs/ca.crt")
CLUSTER_WSS_URL = os.getenv("CLUSTER_WSS_URL", "wss://localhost/ws")

@app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    logger.info("Local UI connected.")
    
    # Setup mTLS SSL Context
    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=CA_PATH)
    ssl_context.load_cert_chain(certfile=CERT_PATH, keyfile=KEY_PATH)
    ssl_context.check_hostname = False # For demo with localhost
    
    try:
        async with websockets.connect(CLUSTER_WSS_URL, ssl=ssl_context) as cluster_ws:
            logger.info("Connected to cluster gateway via mTLS.")
            
            async def forward_to_cluster():
                try:
                    while True:
                        data = await websocket.receive_text()
                        logger.info("Forwarding telemetry to cluster.")
                        await cluster_ws.send(data)
                except WebSocketDisconnect:
                    pass

            async def forward_to_ui():
                try:
                    while True:
                        response = await cluster_ws.recv()
                        await websocket.send_text(response)
                except websockets.exceptions.ConnectionClosed:
                    pass

            await asyncio.gather(forward_to_cluster(), forward_to_ui())
            
    except Exception as e:
        logger.error(f"Failed to connect to cluster: {e}")
        await websocket.send_text(json.dumps({"status": "error", "detail": f"Cluster connection failed: {str(e)}"}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("edge_server:app", host="0.0.0.0", port=8081, reload=False)
