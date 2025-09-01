# run.ps1 — One-click HTTPS launcher for FastAPI/YOLO app (PowerShell 7+)

param(
  [int]$Port = 8000,
  [switch]$ForceRegen,
  [string[]]$ExtraSANs
)

function Info($m){ Write-Host "[i] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[✓] $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Err($m){ Write-Host "[x] $m" -ForegroundColor Red }

# --- Require PowerShell 7+ ---
if ($PSVersionTable.PSVersion.Major -lt 7) {
  Err "This script needs PowerShell 7+. Run it with 'pwsh'."
  exit 1
}

$root = $PSScriptRoot
$certPem = Join-Path $root "cert.pem"
$keyPem  = Join-Path $root "key.pem"

# --- Resolve Python (prefer .venv, else py -3, else python) ---
function Resolve-Python {
  $venvPy = Join-Path $root ".venv\Scripts\python.exe"
  if (Test-Path $venvPy) { return $venvPy }
  if (Get-Command py -ErrorAction SilentlyContinue) { return "py -3" }
  if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
  return $null
}
$PY = Resolve-Python
if (-not $PY) {
  Err "No Python found on PATH."
  Info "Install with: winget install -e --id Python.Python.3.11"
  exit 1
}

# --- Ensure a local venv ---
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Info "Creating local virtualenv .venv ..."
  if ($PY -eq "py -3") { & py -3 -m venv (Join-Path $root ".venv") } else { & $PY -m venv (Join-Path $root ".venv") }
  if (-not (Test-Path $venvPy)) { Err "Failed to create .venv"; exit 1 }
}
$PY = $venvPy
Ok "Using Python: $PY"

# --- Install packages ---
$packages = @(
  'uvicorn[standard]', 'fastapi', 'jinja2', 'aiohttp', 'aiortc',
  'opencv-python', 'ultralytics', 'pillow', 'starlette', 'websockets'
)
Info "Ensuring Python packages are installed ..."
& $PY -m pip install --upgrade pip > $null
& $PY -m pip install --upgrade @packages

# --- Discover active IPv4s ---
$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object {
    $_.ValidLifetime -ne 0 -and
    $_.IPAddress -ne '127.0.0.1' -and
    -not ($_.IPAddress -like '169.254.*') -and
    ($_.PrefixOrigin -in @('Dhcp','Manual'))
  } | Select-Object -ExpandProperty IPAddress -Unique

$hostname = [System.Net.Dns]::GetHostName()
$sanList = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
$null = $sanList.Add($hostname)
foreach ($ip in $ips) { $null = $sanList.Add($ip) }
if ($ExtraSANs) { foreach ($x in $ExtraSANs) { if ($x) { $null = $sanList.Add($x) } } }

if ($sanList.Count -eq 0) {
  Warn "No active IPv4s detected; continuing with hostname only: $hostname"
  $null = $sanList.Add($hostname)
}

Ok ("SANs: " + ($sanList -join ", "))

# --- Create/refresh cert ---
$needCert = $ForceRegen -or -not (Test-Path $certPem) -or -not (Test-Path $keyPem)
if ($needCert) {
  Info "Generating self-signed certificate ..."
  Remove-Item $certPem, $keyPem -ErrorAction SilentlyContinue

  $cert = New-SelfSignedCertificate `
    -DnsName ($sanList.ToArray()) `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -KeyAlgorithm RSA -KeyLength 2048 `
    -KeyExportPolicy Exportable `
    -NotAfter (Get-Date).AddYears(1)

  if (-not $cert) { Err "Failed to create certificate."; exit 1 }

  # Export CERT -> PEM
  $der = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
  $b64 = [Convert]::ToBase64String($der)
  $pemCert = "-----BEGIN CERTIFICATE-----`n$($b64 -replace '.{64}', '$&`n')`n-----END CERTIFICATE-----`n"
  Set-Content -Path $certPem -Value $pemCert -NoNewline

  # Export KEY -> PEM
  $rsa = $cert.GetRSAPrivateKey()
  $pkcs8 = $rsa.ExportPkcs8PrivateKey()
  $b64k = [Convert]::ToBase64String($pkcs8)
  $pemKey = "-----BEGIN PRIVATE KEY-----`n$($b64k -replace '.{64}', '$&`n')`n-----END PRIVATE KEY-----`n"
  Set-Content -Path $keyPem -Value $pemKey -NoNewline

  Ok "Wrote cert.pem/key.pem"
} else {
  Ok "Found existing cert.pem/key.pem — reusing."
}

# --- Open firewall ---
$ruleName = "Uvicorn $Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
  Info "Adding firewall rule for TCP $Port ..."
  New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Any | Out-Null
  Ok "Firewall rule added."
} else {
  Ok "Firewall rule '$ruleName' already exists."
}

# --- Print all reachable URLs ---
Write-Host ""
Write-Host "Open these on your PHONE (same Wi-Fi or Hotspot):" -ForegroundColor Cyan
foreach ($ip in $ips) {
  Write-Host ("  https://{0}:{1}/sender" -f $ip, $Port) -ForegroundColor Green
}
Write-Host ""

# --- Start Uvicorn ---
& $PY -m uvicorn "server:app" --host 0.0.0.0 --port $Port --ssl-keyfile $keyPem --ssl-certfile $certPem
