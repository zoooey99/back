from flask import Flask, request, send_file, jsonify
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import os
from pathlib import Path
import torch
import gc
import logging
import psutil
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

class Config:
    OUTPUT_DIR = Path("generated_music")
    MODEL_NAME = 'facebook/musicgen-small'
    DEFAULT_DURATION = 30
    DEVICE = 'cpu'  # Default to CPU
    OFFLOAD = True  # Whether to offload model between generations
    USE_FP16 = True  # Enable half precision

Config.OUTPUT_DIR.mkdir(exist_ok=True)

class ModelManager:
    def __init__(self):
        self.model = None
        self.device = Config.DEVICE
        
    def load_model(self):
        if self.model is None:
            logger.info("Loading model...")
            self.model = MusicGen.get_pretrained(Config.MODEL_NAME)
            
            if torch.cuda.is_available() and self.device == 'cuda':
                self.model.to(self.device)
                
                # Convert to half precision if enabled
                if Config.USE_FP16:
                    logger.info("Converting model to FP16")
                    self.model.half()  # Convert to FP16
                    # Ensure the compression model stays in FP32
                    if hasattr(self.model, 'compression_model'):
                        self.model.compression_model.float()
            
            self.model.set_generation_params(
                use_sampling=True,
                temperature=1.0
            )
        return self.model
    
    def unload_model(self):
        if self.model is not None:
            self.model.cpu()
            if Config.USE_FP16:
                self.model.float()  # Convert back to float32 before unloading
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model unloaded")

    def generate(self, prompt, duration):
        try:
            model = self.load_model()
            model.set_generation_params(duration=duration)
            
            # Generate in inference mode to save memory
            with torch.inference_mode():
                if Config.USE_FP16 and torch.cuda.is_available():
                    # Ensure input is in FP16 if model is in FP16
                    audio = model.generate([prompt])
                else:
                    audio = model.generate([prompt])
            
            # Convert back to float32 for audio processing
            audio = audio.float()
            
            if Config.OFFLOAD:
                self.unload_model()
                
            return audio
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            if Config.OFFLOAD:
                self.unload_model()
            raise

model_manager = ModelManager()

def clean_memory():
    """Aggressive memory cleaning"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
def get_memory_usage():
    """Get current memory usage"""
    memory_info = {
        'ram_percent': psutil.Process(os.getpid()).memory_percent(),
        'ram_used_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    }
    if torch.cuda.is_available():
        memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    return memory_info

@app.route('/generate-music', methods=['POST'])
def generate_music():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        prompt = data['prompt']
        duration = data.get('duration', Config.DEFAULT_DURATION)
        
        # Clean memory before generation
        clean_memory()
        
        # Generate unique filename
        timestamp = int(time.time())
        output_path = Config.OUTPUT_DIR / f"generated_music_{timestamp}.wav"
        
        # Generate audio
        start_time = time.time()
        audio = model_manager.generate(prompt, duration)
        
        # Move to CPU and save
        audio = audio.cpu()
        audio_write(
            str(output_path),
            audio.squeeze(0),
            model_manager.load_model().sample_rate,
            strategy="loudness"
        )
        
        # Clean up memory after generation
        clean_memory()
        
        generation_time = time.time() - start_time
        logger.info(f"Generation took {generation_time:.2f} seconds")
        
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=output_path.name
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during music generation'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    memory_info = get_memory_usage()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.model is not None,
        'memory_info': memory_info,
        'fp16_enabled': Config.USE_FP16,
        'device': Config.DEVICE
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Endpoint to force memory cleanup"""
    model_manager.unload_model()
    clean_memory()
    return jsonify({'status': 'cleanup completed', 'memory_info': get_memory_usage()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Initialize on CPU by default
    Config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.run(host='0.0.0.0', port=port)