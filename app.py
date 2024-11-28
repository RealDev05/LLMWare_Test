from flask import Flask, request, jsonify
from llmware.models import ModelCatalog
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


class PromptService:
    def __init__(self, model_name='llmware/bling-1.4b-0.1'):
        try:
            # Load the specified model
            self.model = ModelCatalog().load_model(model_name)
            logging.info(f"Loaded model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def generate_response(self, prompt, max_tokens=200):
        try:
            # Generate response using the loaded model
            response = self.model.inference(prompt, max_new_tokens=max_tokens)
            return {
                "status": "success",
                "prompt": prompt,
                "response": response,
                "model": self.model.model_name
            }
        except Exception as e:
            logging.error(f"Prompt generation error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# Global prompt service instance
prompt_service = PromptService()


@app.route('/generate', methods=['POST'])
def generate_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 200)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    result = prompt_service.generate_response(prompt, max_tokens)
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
