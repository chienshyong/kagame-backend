# Kagame Backend
Reflect your style

## Building and running
Use virtual environment to install requirements with:
- `python3 -m venv .venv`
- `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix/maxOS)
- `pip install -r "requirements.txt"` (Be patient, it'll take some time)
  
Keys:
- Make a copy of keys-example.py and rename it to keys.py
- Put our API keys there. If everything goes right, changes should be ignored so keys.py is never pushed 

Run for development with:
- `uvicorn main:app --reload`

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