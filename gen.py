from flask import Flask, request, send_file, jsonify, Response, stream_with_context
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import os
from pathlib import Path
import torch
import gc
import logging
import threading
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

class Config:
    OUTPUT_DIR = Path("generated_music")
    MODEL_NAME = 'facebook/musicgen-small'
    DEFAULT_DURATION = 30
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OFFLOAD = True
    USE_FP16 = True
    CHUNK_SIZE = 8192

Config.OUTPUT_DIR.mkdir(exist_ok=True)

class ModelManager:
    def __init__(self):
        self.model = None
        self.device = Config.DEVICE
        self._loading = False
        self._ready = False
        self._loading_thread = None
        # Start loading in background
        self.start_loading()
        
    def start_loading(self):
        def load_in_background():
            try:
                logger.info("Starting model load in background...")
                self.load_model()
                self._ready = True
                logger.info("Model loaded and ready!")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._ready = False
                self._loading = False
                
        self._loading_thread = threading.Thread(target=load_in_background)
        self._loading_thread.daemon = True
        self._loading_thread.start()
        
    def load_model(self):
        if self.model is None and not self._loading:
            try:
                self._loading = True
                logger.info("Loading model...")
                self.model = MusicGen.get_pretrained(Config.MODEL_NAME)
                
                if torch.cuda.is_available() and self.device == 'cuda':
                    logger.info("Moving model to GPU...")
                    self.model.to(self.device)
                    
                    if Config.USE_FP16:
                        logger.info("Converting model to FP16")
                        self.model.half()
                        if hasattr(self.model, 'compression_model'):
                            self.model.compression_model.float()
                
                self.model.set_generation_params(
                    use_sampling=True,
                    temperature=1.0
                )
                logger.info("Model loaded successfully")
            finally:
                self._loading = False
        return self.model

    def is_ready(self):
        return self._ready and self.model is not None
    
    def unload_model(self):
        if self.model is not None:
            self.model.cpu()
            if Config.USE_FP16:
                self.model.float()
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self._ready = False
            logger.info("Model unloaded")

    def generate(self, prompt, duration):
        if not self.is_ready():
            raise RuntimeError("Model is not ready yet")
            
        try:
            model = self.load_model()
            model.set_generation_params(duration=duration)
            
            with torch.inference_mode():
                if Config.USE_FP16 and torch.cuda.is_available():
                    audio = model.generate([prompt])
                else:
                    audio = model.generate([prompt])
            
            audio = audio.float()
            
            if Config.OFFLOAD:
                self.unload_model()
                
            return audio
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            if Config.OFFLOAD:
                self.unload_model()
            raise

# Initialize model manager globally
logger.info("Initializing ModelManager...")
model_manager = ModelManager()

@app.route('/ready', methods=['GET'])
def ready_check():
    """Endpoint for checking if the model is loaded and ready"""
    if model_manager.is_ready():
        return jsonify({'status': 'ready', 'message': 'Model is loaded and ready'})
    else:
        return jsonify({
            'status': 'loading',
            'message': 'Model is still loading'
        }), 503  # Service Unavailable

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check that doesn't depend on model loading"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'ready' if model_manager.is_ready() else 'loading',
        'device': Config.DEVICE
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)