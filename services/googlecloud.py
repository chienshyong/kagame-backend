from google.cloud import storage
from google.auth import load_credentials_from_file
from services import SERVICE_ACCOUNT_JSON_PATH

BUCKET_NAME = "kagame_bucket_1"

credentials, project = load_credentials_from_file(SERVICE_ACCOUNT_JSON_PATH)
storage_client = storage.Client(credentials=credentials, project=project)
bucket = storage_client.bucket("kagame_bucket_1")
