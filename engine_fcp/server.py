import grpc
from concurrent import futures
import time
import os
import torch
import torch.nn as nn
import vap_inference_pb2
import vap_inference_pb2_grpc
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

class AdvancedBKIClassifier(nn.Module):
    def __init__(self, input_dim=6):
        super(AdvancedBKIClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.feature_extractor(x))

class VAPPredictorServicer(vap_inference_pb2_grpc.VAPPredictorServicer):
    def __init__(self):
        # Force cuda:0 since Docker maps isolated GPU to index 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AdvancedBKIClassifier(input_dim=6).to(self.device)
        
        weight_path = "/model/bki_classifier_advanced_cnn.pt"
        if os.path.exists(weight_path):
            logging.info(f"Loading weights from {weight_path} onto {self.device}")
            try:
                checkpoint = torch.load(weight_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model = checkpoint.to(self.device)
                self.model.eval()
                logging.info("Successfully loaded weights.")
            except Exception as e:
                logging.error(f"Failed to load weights: {e}")
        else:
            logging.warning(f"Weight file not found at {weight_path}. Using uninitialized model.")
            self.model.eval()

    def Predict(self, request, context):
        try:
            input_vector = torch.tensor([[
                request.peep,
                request.pip,
                request.fio2,
                request.hrv,
                request.procalcitonin,
                request.p_f_ratio
            ]], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                risk_tensor = self.model(input_vector)
                risk_probability = float(risk_tensor.cpu().item())
            
            return vap_inference_pb2.InferenceResponse(risk_probability=risk_probability)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vap_inference_pb2.InferenceResponse()

def serve():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.1, 0)
        except Exception as e:
            logging.warning(f"Could not set memory fraction: {e}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vap_inference_pb2_grpc.add_VAPPredictorServicer_to_server(VAPPredictorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("FCP-Former gRPC Server started on port 50051.")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
