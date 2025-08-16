import uvicorn
import os

if __name__ == "__main__":
    # Usar la variable PORT de Cloud Run o 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Sin reload automático para evitar problemas
        workers=1,  # Un solo worker para evitar conflictos de memoria
        loop="asyncio",  # Usar asyncio para mejor rendimiento
        http="httptools",  # Usar httptools para mejor rendimiento HTTP
        access_log=False,  # Deshabilitar logs de acceso para mejor rendimiento
        log_level="info"  # Solo logs importantes
    ) 