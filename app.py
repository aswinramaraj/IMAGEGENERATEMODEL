from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from diffusers import StableDiffusionXLPipeline
from io import BytesIO
import base64
import torch

app = FastAPI()

# Load the model on CPU
print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True
).to("cpu")
print("✅ Model loaded")

class PromptRequest(BaseModel):
    prompt: str
    response_format: str = "b64_json"
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"

@app.post("/generate")
def generate_image(data: PromptRequest):
    try:
        prompt = data.prompt
        image = pipe(prompt=prompt).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "data": [{
                "b64_json": img_base64
            }]
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate image: {str(e)}"}
        )

@app.get("/")
def home():
    return {"message": "API is working"}
