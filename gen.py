from flask import Flask, request, send_file, jsonify
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import os
from pathlib import Path

app = Flask(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("generated_music")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize the model globally to avoid reloading it with each request
print("Loading MusicGen model...")
model = MusicGen.get_pretrained('facebook/musicgen-small')

@app.route('/generate-music', methods=['POST'])
def generate_music():
    try:
        # Get parameters from request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        prompt = data['prompt']
        duration = data.get('duration', 30)  # Default 10 seconds if not specified
        
        # Generate unique filename
        timestamp = int(time.time())
        output_path = OUTPUT_DIR / f"generated_music_{timestamp}.wav"
        
        # Set generation parameters
        model.set_generation_params(
            use_sampling=True,
            temperature=1.0,
            duration=duration
        )
        
        print(f"Generating music for prompt: '{prompt}'")
        
        # Generate the audio
        start_time = time.time()
        audio = model.generate([prompt])
        generation_time = time.time() - start_time
        
        # Save the audio
        audio_write(
            str(output_path),
            audio.squeeze(0).cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Generation took {generation_time:.2f} seconds")
        
        # Return the audio file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=output_path.name
        )
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during music generation'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'facebook/musicgen-small'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)