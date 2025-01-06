from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Dict
import torch
from torchvision import transforms
from timeit import default_timer as timer

import torchvision

# Load the TorchScript model
MODEL_PATH = "./models/effnetb2_scripted.pt"
model = torch.jit.load(MODEL_PATH)
model.eval()

# Define image transformations (use the same as your Gradio implementation)
effnetb2_transforms = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
# Class names (update with your actual class names)
class_names = ["Barbery", "Harry"]

# Initialize FastAPI app
app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Start timer
        start_time = timer()

        # Load the uploaded image
        img = Image.open(file.file).convert("RGB")

        # Apply transformations
        input_tensor = effnetb2_transforms(img).unsqueeze(0)

        # Perform inference
        with torch.inference_mode():
            pred_probs = torch.softmax(model(input_tensor), dim=1)

        # Create a prediction dictionary
        pred_labels_and_probs: Dict[str, float] = {
            class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
        }

        # Calculate inference time
        pred_time = round(timer() - start_time, 5)

        # Return the response
        return JSONResponse({
            "predictions": pred_labels_and_probs,
            "inference_time": pred_time
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
