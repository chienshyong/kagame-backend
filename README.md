# Kagame Backend
Reflect your style

## Building and running
Google Cloud Storage uses Application Default Credentials, not API keys. This means you're required to install **Google Cloud SDK Shell**. Skip this section you already have it. Otherwise, here are the installation steps:
- Install gcloud CLI https://cloud.google.com/sdk/docs/install
- Then in terminal do:
- `gcloud init`
- `gcloud auth application-default login`
- Your browser should open. Login with Kagame's gmail account. When successful, you should get something like:
- `Credentials saved to file: [some\local\filepath\gcloud\application_default_credentials.json]`

Use Python virtual environment to install the requirements:
- `python3 -m venv .venv`
- `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix/maxOS)
- `pip install -r "requirements.txt"` (Be patient, it'll take some time)
  
Set API Keys variable values:
- Make a copy of keys-example.py and rename it to keys.py
- Put our API keys there. You can find this in our groupchat. If everything goes right, changes should be ignored so keys.py is never pushed 

Test if everything works:
- Run `fashion_clip_test.ipynb`, `db_test.ipynb` and `gcloud_test.ipynb`. See if errors popup

Run backend for development with:
- `uvicorn main:app --reload`
- `uvicorn main:app` If you don't want server to auto-reload everytime you make a change.

## Routes
Include all routes in the `/routes` folder, separated by functionality. Include `auth.py` in almost all routes to use for protecting endpoints and identifying the user with `get_current_user()`, which examines the http headers for an auth token.

## Database connection
`kagameDB.py` connects to the database, and can be used from anywhere by importing it.

Store and get images with `services/image.py`. For now, it converts the image into binary data that can be stored in mongo. When we change to a file system just need to change this service.

## Deployment
Deployment to runpod - Build docker image with:
- `docker build -t kagame-backend .`
- `docker run -d -p 80:80 kagame-backend`

Push to docker hub:
- `docker login -u kagameteam`
- Password: see in chat
- `docker tag kagame-backend kagameteam/kagame-backend:latest`
- `docker push kagameteam/kagame-backend:latest`

Use kagameteam/kagame-backend as the image on the runpod container.