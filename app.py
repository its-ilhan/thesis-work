from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from predict import process_single_audio

app = FastAPI()

# Allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # 1. Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # 2. Run your inference pipeline
        ai_probability = process_single_audio(temp_file_path)
        
        # 3. Format the result
        is_human = ai_probability < 0.5
        confidence = (1 - ai_probability) if is_human else ai_probability
        
        return JSONResponse({
            "status": "success",
            "classification": "Human" if is_human else "AI Generated",
            "confidence": round(confidence * 100, 2),
            "ai_probability_score": round(ai_probability, 4)
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
        
    finally:
        # Clean up the temporary file so your server doesn't run out of storage
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Run this script using: uvicorn app:app --reload