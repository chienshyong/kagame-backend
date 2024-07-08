# Kagame Backend
Reflect your style

## MongoDB connection
`./keys/` folder contains connection strings / api keys. Replace the placeholder values with the key given in the group chat. For security NEVER push the API key to github even if it's a private repository. 

## Building and running
Run for development with:
- `uvicorn main:app --reload`

Build docker image with:
- `docker build -t kagame-backend .`
- `docker run -d -p 80:80 kagame-backend`

Push to docker hub:
- `docker login -u kagameteam`
- Password: see in chat
- `docker tag kagame-backend kagameteam/kagame-backend:latest`
- `docker push kagameteam/kagame-backend:latest`

Use kagameteam/kagame-backend as the image on the runpod container.