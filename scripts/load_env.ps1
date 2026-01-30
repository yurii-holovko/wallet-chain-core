param(
    [string]$EnvFile = ".env"
)

if (-not (Test-Path $EnvFile)) {
    Write-Error "Env file not found: $EnvFile"
    exit 1
}

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) {
        return
    }
    $parts = $line.Split("=", 2)
    if ($parts.Length -lt 2) {
        return
    }
    [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
}

Write-Host "Loaded environment variables from $EnvFile"
