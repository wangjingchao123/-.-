$ErrorActionPreference = 'Stop'

# 仅供受信任的同一 Wi-Fi 或校园网使用。不要在公共网络中开放本服务。
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = if ($env:CAR_AI_AGENT_PORT) { $env:CAR_AI_AGENT_PORT } else { '5001' }
$env:CAR_AI_AGENT_HOST = '0.0.0.0'
$env:CAR_AI_AGENT_PORT = $port

$ip = Get-CimInstance Win32_NetworkAdapterConfiguration |
    Where-Object { $_.IPEnabled -and $_.IPAddress } |
    ForEach-Object { $_.IPAddress } |
    Where-Object { $_ -match '^192\.168\.|^10\.|^172\.(1[6-9]|2[0-9]|3[0-1])\.' } |
    Select-Object -First 1

Write-Host ''
Write-Host '车醒瞳安局域网服务正在启动' -ForegroundColor Cyan
if ($ip) {
    Write-Host "同一局域网设备可先访问：http://$ip`:$port" -ForegroundColor Green
    Write-Host '提示：非 localhost 的 HTTP 页面在部分浏览器中不能调用摄像头；需要 HTTPS 后才可保证其他电脑实时摄像头可用。' -ForegroundColor Yellow
} else {
    Write-Host '未自动识别局域网 IP。请执行 ipconfig，使用 IPv4 地址和端口访问。' -ForegroundColor Yellow
}
Write-Host '按 Ctrl+C 停止服务。' -ForegroundColor DarkGray

& python (Join-Path $projectRoot 'web_app.py')
