from fastapi import HTTPException, APIRouter, Depends, File, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from services.remove_bg import remove_bg
from services.image import store_blob, get_blob_url, SHORT_EXPIRY

router = APIRouter()

'''
POST /image/remove-bg -> Removes background from the image. Output is less reliable with multiple objects
POST /image/upload-image -> Uploads an image to google cloud storage. Returns the randomly generated filename of the inserted image.
POST /image/get-image -> Fetches an image from google cloud storage by filename. Returns a signed URL that expires in 5 seconds.
'''


@router.post("/image/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()  # Check if the file is an actual image
        image = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    # Image is good
    output = remove_bg(image)

    # Output the file as a stream because google says it's faster.
    buffer = BytesIO()
    output.save(buffer, format="png")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@router.post("/image/upload-image")  # TODO: Remove this because this is an unprotected image upload.
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()  # Check if the file is an actual image
        image = Image.open(BytesIO(contents))  # Re-open to handle potential truncation issue
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    image_name = store_blob(contents, f"image/{image.format}")
    return {"image_name": image_name}


@router.get("/image/get-image")  # TODO: Remove this because this is an unprotected image access.
async def get_image(image_name: str):
    url = get_blob_url(image_name, SHORT_EXPIRY)
    if url == None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image_name"
        )
    return {"image_url": url}
