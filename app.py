import io
import ssl
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz  
import easyocr
import numpy as np
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
reader = easyocr.Reader(['en'])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows connections while working locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        all_text = []
        
        if file.content_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_bytes))
            img_np = np.array(image)
            result = reader.readtext(img_np, detail=0)
            if result: all_text.extend(result)
        else:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                img_np = np.array(image)
                result = reader.readtext(img_np, detail=0)
                if result: all_text.extend(result)
        
        return {"status": "success", "text": all_text if all_text else ["[No text found]"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)