from fastapi import FastAPI
from routes import catalogue_routes, image_routes, user_routes, wardrobe_routes, profile_routes

#Run with: uvicorn main:app --reload
app = FastAPI()

app.include_router(user_routes.router)
app.include_router(catalogue_routes.router)
app.include_router(wardrobe_routes.router)
app.include_router(image_routes.router)
app.include_router(profile_routes.router)

@app.get("/")
def hello():
    return {"message": "Hello, World!"}