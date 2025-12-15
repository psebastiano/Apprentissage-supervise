param(
    [Parameter(Mandatory=$true)]
    [string]$PythonScriptPath,

    [Parameter(Mandatory=$false)]
    [string]$LogFile = "log_clean.txt"
)

# Verifier si le fichier Python existe
if (-not (Test-Path $PythonScriptPath)) {
    Write-Host "Erreur : Le fichier Python specifie n'existe pas a '$PythonScriptPath'" -ForegroundColor Red
    exit 1
}

# Assurer que le repertoire du log existe
$LogDir = Split-Path -Path $LogFile -Parent
if ($LogDir -ne "" -and -not (Test-Path $LogDir)) {
    New-Item -Path $LogDir -ItemType Directory | Out-Null
    Write-Host "Creation du repertoire de logs : $LogDir" -ForegroundColor Yellow
}

# --- Set environment variables for UTF-8 encoding ---
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host "Execution de '$PythonScriptPath'..." -ForegroundColor Cyan

# --- 1. Execution de Python et capture de la sortie brute ---
$rawOutput = python $PythonScriptPath 2>&1

# --- 2. Definition du filtre de nettoyage ---
$replacements = @{
    "\u03b7" = "eta"   # Caractere grec 'eta'
    "\u2705" = "[OK]"  # Emoji checkmark
    "\u274c" = "[ERR]" # Emoji cross mark
}

# --- 3. Nettoyage de la sortie ---
$cleanOutput = @()
foreach ($line in $rawOutput) {
    $cleanedLine = $line.ToString()
    
    foreach ($char in $replacements.GetEnumerator()) {
        $cleanedLine = $cleanedLine -replace $char.Name, $char.Value
    }
    
    $cleanOutput += $cleanedLine
}

# --- 4. Ecriture du log nettoye ---
$cleanOutput | Out-File -FilePath $LogFile -Encoding UTF8

# --- Affichage des resultats (sans caracteres d'indentation complexes) ---

Write-Host "`nExecution terminee." -ForegroundColor Green
Write-Host "Sortie nettoyee et enregistree dans le fichier : $LogFile" -ForegroundColor Green

# Afficher les informations de sortie
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nAvertissement : Le script Python a pu se terminer avec un code d'erreur non nul ($LASTEXITCODE)." -ForegroundColor Red
}