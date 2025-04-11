# KagaMe Backend

The backend code for KagaMe, a virtual stylist and digital wardrobe app.

The frontend code can be found [here](https://github.com/chienshyong/kagame_app).

## Building and Running
Use Python virtual environment to install the requirements:
- `python3 -m venv .venv`
- `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix/maxOS)
- `pip install -r "requirements.txt"`

Run backend for development with:
- `uvicorn main:app --reload`
- `uvicorn main:app` If you don't want server to auto-reload everytime you make a change.

Google Cloud Storage uses Application Default Credentials, not API keys. This means you're required to install **Google Cloud SDK Shell**. Skip this section you already have it. Otherwise, here are the installation steps:
- Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Then in terminal do:
- `gcloud init`
- `gcloud auth application-default login`
- Your browser should open. Login with Kagame's gmail account. When successful, you should get something like:
- `Credentials saved to file: [some\local\filepath\gcloud\application_default_credentials.json]`

Set API Keys variable values and other secrets:
- Make a copy of the `secretstuff-example` folder and rename it to `secretstuff`
- Populate the secrets with content in our groupchat. If everything goes right, changes should be ignored so keys.py is never pushed 

Test if everything works:
- Run `fashion_clip_test.ipynb`, `db_test.ipynb` and `gcloud_test.ipynb`. See if errors popup.
TODO(maybe?): Combine all the test Python Notebook into one file

## Deployment
Deployment to runpod - Build docker image with:
- `docker build -t kagame-backend .`
- `docker run -d -p 80:80 kagame-backend`

Push to docker hub:
- `docker login -u kagameteam`
- Password: see in chat
- `docker tag kagame-backend kagameteam/kagame-backend:latest`
- `docker push kagameteam/kagame-backend:latest`

Use image kagameteam/kagame-backend on the cloud server.

## Database connection
`/services/mongodb.py` connects to the MongoDB database, and can be used from anywhere by importing it.

`/services/image.py` connects to the Google Cloud bucket for storing images and generate temporary image urls to them.

## Routes
All routes are included in the `/routes` folder, separated by functionality (Catalogue, Image, Profile, User, and Wardrobe).

Code to authenticate and identify the user is done with `/services/user.py` . The middleware `get_current_user()` is used on almost all routes to examine the http header for an auth token that identifies the user.

Code to query the OpenAI API for GPT-4o can be found in `/services/openai.py`.