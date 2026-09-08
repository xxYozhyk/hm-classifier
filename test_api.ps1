# test_api.ps1
# Устанавливаем кодировку UTF-8 для корректного отображения русского текста
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 > $null

$baseUrl = "http://localhost:8000"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ТЕСТИРОВАНИЕ H&M CLASSIFIER API" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Тест GET /
Write-Host "1. Тест GET /" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "✅ Статус: 200 OK" -ForegroundColor Green
    Write-Host "   Сервис: $($response.service)" -ForegroundColor White
    Write-Host "   Версия: $($response.version)" -ForegroundColor White
    Write-Host "   Модель: $($response.model_used)" -ForegroundColor White
} catch {
    Write-Host "❌ Ошибка: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# 2. Тест GET /health
Write-Host "2. Тест GET /health" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✅ Статус: 200 OK" -ForegroundColor Green
    Write-Host "   Статус: $($response.status)" -ForegroundColor White
    Write-Host "   Модель загружена: $($response.model_loaded)" -ForegroundColor White
    Write-Host "   Классы: $($response.classes -join ', ')" -ForegroundColor White
} catch {
    Write-Host "❌ Ошибка: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# 3. Тест POST /predict - корректные запросы
Write-Host "3. Тест POST /predict (корректные запросы)" -ForegroundColor Yellow

$testCases = @(
    @{ prod_name = "Strap top"; detail_desc = "Jersey top with narrow shoulder straps" },
    @{ prod_name = "Socks"; detail_desc = "Fine-knit socks in cotton blend" },
    @{ prod_name = "Jeans"; detail_desc = "5-pocket jeans in washed denim" }
)

foreach ($test in $testCases) {
    try {
        $body = $test | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$baseUrl/predict" -Method Post -Body $body -ContentType "application/json"
        $confidence = $response.probabilities.$($response.predicted_group) * 100
        Write-Host "✅ $($test.prod_name): $($response.predicted_group)" -ForegroundColor Green
        Write-Host "   Уверенность: $([math]::Round($confidence, 2))%" -ForegroundColor White
    } catch {
        Write-Host "❌ $($test.prod_name): Ошибка" -ForegroundColor Red
    }
}
Write-Host ""

# 4. Тест POST /predict - некорректные запросы
Write-Host "4. Тест POST /predict (некорректные запросы)" -ForegroundColor Yellow

$invalidTests = @(
    @{ prod_name = ""; detail_desc = "Description without name" },
    @{ prod_name = "Name without description"; detail_desc = "" },
    @{ prod_name = ""; detail_desc = "" }
)

foreach ($test in $invalidTests) {
    try {
        $body = $test | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$baseUrl/predict" -Method Post -Body $body -ContentType "application/json"
        Write-Host "⚠️ Пустой текст не вызвал ошибку (ожидалась 400)" -ForegroundColor Yellow
    } catch {
        if ($_.Exception.Response.StatusCode -eq 400) {
            Write-Host "✅ Ожидаемая ошибка 400" -ForegroundColor Green
        } else {
            Write-Host "❌ Неожиданная ошибка: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ТЕСТИРОВАНИЕ ЗАВЕРШЕНО" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan