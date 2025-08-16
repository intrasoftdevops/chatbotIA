import uvicorn
import os

if __name__ == "__main__":
    # Usar la variable PORT de Cloud Run o 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    ) 