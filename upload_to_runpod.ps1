# PowerShell script to upload files to Runpod
# Usage: Right-click → "Run with PowerShell"

$RUNPOD_HOST = "213.173.108.139"
$RUNPOD_PORT = "11527"
$RUNPOD_USER = "root"
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519"
$REMOTE_PATH = "/workspace/"

Write-Host "Uploading files to Runpod..." -ForegroundColor Cyan

# Check if OpenSSH is available
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: OpenSSH not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please use one of these options:" -ForegroundColor Yellow
    Write-Host "1. Install WinSCP: https://winscp.net/eng/download.php" -ForegroundColor Cyan
    Write-Host "2. Install Git Bash and run from there" -ForegroundColor Cyan
    Write-Host "3. Enable OpenSSH client in Windows Features" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "For WinSCP, use these settings:" -ForegroundColor Yellow
    Write-Host "  Host: $RUNPOD_HOST"
    Write-Host "  Port: $RUNPOD_PORT"
    Write-Host "  User: $RUNPOD_USER"
    Write-Host "  Key:  $SSH_KEY"
    pause
    exit 1
}

# Upload files
scp -P $RUNPOD_PORT -i $SSH_KEY smart_rag.py knowledge.json server.py .env.runpod "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_PATH}"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Upload successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now connecting to Runpod..." -ForegroundColor Cyan
    Write-Host ""

    # SSH connect and setup
    ssh -p $PORT -i $KEY "${USER}@${HOST}" @"
cd /workspace
mv .env.runpod .env
pip install fastapi uvicorn faster-whisper openai elevenlabs python-dotenv scipy numpy
echo ""
echo "Setup complete! Starting server..."
python server.py
"@
} else {
    Write-Host "Upload failed!" -ForegroundColor Red
    Write-Host "Make sure PowerShell is running and you have SSH access."
}
