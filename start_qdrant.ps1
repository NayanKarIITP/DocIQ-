
Write-Host "Starting Qdrant vector database..." -ForegroundColor Green
Write-Host "Qdrant runs on http://localhost:6333"
Write-Host "Keep this window open."
& "C:\Users\nkar9\Downloads\multimodal-rag\multimodal-rag\qdrant_local\qdrant.exe" --storage-path ./qdrant_storage
