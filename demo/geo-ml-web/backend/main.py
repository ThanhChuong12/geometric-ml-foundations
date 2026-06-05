from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.part1_routes import router as part1_router
from api.part2_routes import router as part2_router

app = FastAPI(
    title="Frame Averaging Demo API",
    description="Backend API for the Frame Averaging vs Data Augmentation demo.",
    version="1.0.0"
)

# Configure CORS so the Next.js frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routes
app.include_router(part1_router, prefix="/api/part1")
app.include_router(part2_router, prefix="/api/part2")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
