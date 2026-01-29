# Local test script for Unum workflow
# This invokes the Hello function locally

Write-Host "Testing Unum Hello-World workflow locally..." -ForegroundColor Green

# Invoke Hello function
Write-Host "`nInvoking Hello function..." -ForegroundColor Yellow
sam local invoke HelloFunction -t template.yaml -e test-event.json --region eu-central-1

Write-Host "`nNote: World function is invoked automatically by Hello," -ForegroundColor Cyan
Write-Host "but SAM local doesn't support Lambda-to-Lambda invocations." -ForegroundColor Cyan
Write-Host "To test World, invoke it manually with Hello's output." -ForegroundColor Cyan
