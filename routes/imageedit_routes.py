from fastapi import HTTPException, APIRouter, Depends, File, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from services.remove_bg import remove_bg

router = APIRouter()

'''
POST /image-edit/remove-bg -> Removes background from the image. Output is less reliable with multiple images
'''

@router.post("/image-edit/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()  # Check if the file is an actual image
        image = Image.open(BytesIO(contents))  # Re-open to handle potential truncation issue
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    # Image is good
    output = remove_bg(image)
    print(f"Removed an image's background")

    # Output the file as a stream because google says it's faster.
    buffer = BytesIO()
    output.save(buffer, format="png")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
