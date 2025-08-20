# Fix PowerShell Execution Policy
Write-Host "🔧 Fixing PowerShell Execution Policy..." -ForegroundColor Yellow

# Check current policy
Write-Host "Current Execution Policy: $(Get-ExecutionPolicy)" -ForegroundColor Cyan

# Set execution policy for current user
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "✅ Execution Policy updated successfully!" -ForegroundColor Green
    Write-Host "New Execution Policy: $(Get-ExecutionPolicy)" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Failed to update execution policy: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nNow you can run npm commands!" -ForegroundColor Green
Write-Host "Try: npm install -g vercel" -ForegroundColor Cyan

# Keep window open
Read-Host "Press Enter to continue..."
