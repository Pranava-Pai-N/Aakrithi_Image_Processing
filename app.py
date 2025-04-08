from fastapi import FastAPI, UploadFile, File
import fitz  
import cv2
import pytesseract
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF and Image to Text Extraction API!"}

@app.post("/extract_text_from_pdf")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    try:
        file_path = f"./{file.filename}"
        if not(file.filename.endswith('.pdf')): 
            return {"error": "File is not a PDF"}
        
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(await file.read())

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        return {"extracted_text": text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/extract_text_from_image")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        file_path = f"./{file.filename}"
        if not(file.filename.endswith('.jpg') or file.filename.endswith('.png') or file.filename.endswith('.jpeg')):
            return {"error": "File is not an image"}
        
        with open(file_path, "wb") as img_file:
            img_file.write(await file.read())

        img = cv2.imread(file_path)
        text = pytesseract.image_to_string(img)
        return {"extracted_text": text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
