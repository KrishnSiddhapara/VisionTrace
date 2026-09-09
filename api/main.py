import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config.settings import settings
from api.routes import router as api_router

app = FastAPI(
    title="VisionTrace AI Video Intelligence Platform API",
    description="Production API for VisionTrace AI - Computer Vision, Spatial Tracking, VLM & Temporal Reasoning",
    version="2.0.0"
)

# Enable CORS for frontend development server (Vite on 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api")

# Mount media directories
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/media/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/media/processed", StaticFiles(directory=str(settings.PROCESSED_DIR)), name="processed")

# Mount web/dist static assets if built
web_dist_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
if web_dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(web_dist_dir / "assets")), name="static_assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("media/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        file_path = web_dist_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(web_dist_dir / "index.html")

@app.get("/")
async def root():
    if web_dist_dir.exists():
        return FileResponse(web_dist_dir / "index.html")
    return {
        "name": "VisionTrace AI API",
        "status": "online",
        "version": "2.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
