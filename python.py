import io
import ssl
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image

# Force disable SSL validation errors while downloading OCR data weights
ssl._create_default_https_context = ssl._create_unverified_context

print("Loading OCR Engine...")
reader = easyocr.Reader(['en'])

app = FastAPI()

# Allow connections from your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skylark14-stack.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        all_extracted_text = []
        
        # Check if the uploaded file is an image or a PDF
        if file.content_type.startswith("image/"):
            # Process standalone images directly through PIL and NumPy
            image = Image.open(io.BytesIO(file_bytes))
            img_np = np.array(image)
            result = reader.readtext(img_np, detail=0)
            if result:
                all_extracted_text.extend(result)
                
        else:
            # Process standard PDFs from memory streams
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page out to high-res imagery matrix
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                img_np = np.array(image)
                
                result = reader.readtext(img_np, detail=0)
                if result:
                    all_extracted_text.extend(result)
        
        if not all_extracted_text:
            return {"status": "success", "text": ["[OCR could not find any text]"]}
            
        return {"status": "success", "text": all_extracted_text}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)