from services.googlecloud import bucket
from PIL import Image
from io import BytesIO
from typing import Optional
import datetime
import base64
import uuid

SHORT_EXPIRY = datetime.timedelta(seconds=15)
DEFAULT_EXPIRY = datetime.timedelta(minutes=1)
LONG_EXPIRY = datetime.timedelta(minutes=15)


def store_blob(data: bytes, content_type: str = "unknown") -> str:
    blob_name = str(uuid.uuid4())
    bucket.blob(blob_name).upload_from_string(data, content_type=content_type)
    return blob_name


def get_blob_url(blob_name: str, expiration: datetime.timedelta) -> Optional[str]:
    # TODO(maybe): Throw exception if blob_name is invalid?
    blob = bucket.blob(blob_name)

    if (not blob.exists()):
        return None

    return blob.generate_signed_url(expiration=expiration, method='GET')
