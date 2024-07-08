# Kagame Backend
Reflect your style

# MongoDB connection
`./keys/` folder contains connection strings / api keys. Replace the placeholder values with the key given in the group chat. For security NEVER push the API key to github even if it's a private repository. 

# Building and running
Run with:
uvicorn main:app --reload

Build docker with:
docker build -t kagame-backend .
docker run -d -p 80:80 kagame-backend