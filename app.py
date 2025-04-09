from fastapi import FastAPI, UploadFile, File
from ModelCode import classify_image
from PIL import Image
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Image Processing and Classification API!"}


@app.post("/Image_Processing")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        file_path = f"./{file.filename}"
        if not(file.filename.endswith('.jpg') or file.filename.endswith('.png') or file.filename.endswith('.jpeg')):
            return {"error": "File is not an image"}
        
        with open(file_path, "wb") as img_file:
            img_file.write(await file.read())
            
        result = classify_image(file_path)
        return {
            "Type":str(result["Type"]),
        }
         
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
