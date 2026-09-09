"""
VisionTrace AI - Professional AI Video Intelligence Platform
Entry point server launcher.
"""
import uvicorn

if __name__ == "__main__":
    print("==========================================================")
    print("Starting VisionTrace AI Platform Server on http://localhost:8000")
    print("==========================================================")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
