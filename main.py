from fastapi import FastAPI
from routes import auth_routes, catalogue_routes, wardrobe_routes, imageedit_routes

#Run with: uvicorn main:app --reload
app = FastAPI()

app.include_router(auth_routes.router)
app.include_router(catalogue_routes.router)
app.include_router(wardrobe_routes.router)
app.include_router(imageedit_routes.router)

@app.get("/")
def hello():
    return {"message": "Hello, World!"}