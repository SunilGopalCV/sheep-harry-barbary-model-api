from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Dict
import torch
from torchvision import transforms
from timeit import default_timer as timer
import os

import torchvision

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Adjust this for security (e.g., only allow your frontend domain)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./models/effnetb2_scripted.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at path: {MODEL_PATH}")

model = torch.jit.load(MODEL_PATH)
model.eval()

effnetb2_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class names for prediction
class_names = ["Barbery", "Harry"]


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Start Timer
        start_time = timer()

        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return JSONResponse({"error": "Invalid file type. Please upload a PNG, JPG, or JPEG image."}, status_code=400)

        # Load the uploaded image
        img = Image.open(file.file).convert("RGB")

        # Apply transformations
        input_tensor = effnetb2_transforms(img).unsqueeze(0)

        # Perform inference
        with torch.inference_mode():
            pred_probs = torch.softmax(model(input_tensor), dim=1)

        # Prepare predictions
        pred_labels_and_probs: Dict[str, float] = {
            class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
        }

        # Calculate inference time
        pred_time = round(timer() - start_time, 5)

        # Return the result as JSON
        return JSONResponse({
            "predictions": pred_labels_and_probs,
            "inference_time": pred_time
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
