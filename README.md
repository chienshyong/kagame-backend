# Kagame Backend
Reflect your style

## Routes
Include all routes in the `/routes` folder, separated by functionality. `auth.py` is included in `/routes` to use for protecting endpoints.

## MongoDB connection
`./keys/` folder contains connection strings / api keys / secrets. Replace the placeholder values with the key given in the group chat. For security NEVER push the API key to github even if it's a private repository. Prevent committing the keys to the repo by running `git update-index --assume-unchanged ./keys/secrets.py ./keys/mongodb.py` for each file.

`kagameDB.py` connects to the database, and can be used from anywhere by importing it.

## Building and running
Install requirements with:
- `pip install -r "requirements.txt"`
  
Run for development with:
- `uvicorn main:app --reload`

Deployment to runpod - Build docker image with:
- `docker build -t kagame-backend .`
- `docker run -d -p 80:80 kagame-backend`

Push to docker hub:
- `docker login -u kagameteam`
- Password: see in chat
- `docker tag kagame-backend kagameteam/kagame-backend:latest`
- `docker push kagameteam/kagame-backend:latest`

Use kagameteam/kagame-backend as the image on the runpod container.