import cv2
import numpy as np
from fastapi import FastAPI, Request
import base64

IMAGE_SIZE = (128, 128)
app = FastAPI()

@app.get("/api")
def read_img ():
    return'API ON'

@app.post("/api/preprocess")
async def read_image(image_data: Request):
    image_dataJson = await image_data.json()
    decoded_imgData = base64.b64decode(image_dataJson["image"])
    
    # แปลง binary data เป็น NumPy array
    image_array = np.frombuffer(decoded_imgData, np.uint8)
    # Decode numpy array to image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    images = np.array(image, dtype = 'float32')
    print(images.shape)
    # เพิ่มมิติแรกเป็น None ให้กลายเป็น (1,128,128,3)
    images = images[np.newaxis, :, :, :]
    print(images.shape)

    return {'img': images.tolist()}
    # return image