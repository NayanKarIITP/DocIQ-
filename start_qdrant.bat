

@echo off
echo Starting Qdrant vector database...
echo Qdrant will run on http://localhost:6333
echo Keep this window open while using the project.
echo.

REM Create storage folder if it doesn't exist
if not exist "qdrant_storage" mkdir qdrant_storage

REM Try new flag first (Qdrant v1.9+), fallback to config file
qdrant_local\qdrant.exe 2>nul
if errorlevel 1 (
    echo Trying with config file...
    echo storage:\n  storage_path: ./qdrant_storage > qdrant_local\config.yaml
    qdrant_local\qdrant.exe --config-path qdrant_local\config.yaml
)