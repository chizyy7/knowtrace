# PowerShell script to download ASSISTments 2009-2010 skill-builder dataset
# Tries Kaggle CLI first, falls back to manual instructions

# Configuration
$RawDataDir = "data\raw"
$DatasetName = "nicolasw/assistments-2009-2010"
$ExpectedFile = "skill_builder_data.csv"

# Ensure raw data directory exists
if (-not (Test-Path $RawDataDir)) {
    New-Item -ItemType Directory -Path $RawDataDir | Out-Null
}

Write-Host "Attempting to download ASSISTments 2009-2010 skill-builder dataset..."

# Try Kaggle CLI
if (Get-Command kaggle -ErrorAction SilentlyContinue) {
    Write-Host "Kaggle CLI found. Attempting download..."
    try {
        kaggle datasets download -d $DatasetName -p $RawDataDir --unzip
        if (Test-Path (Join-Path $RawDataDir $ExpectedFile)) {
            Write-Host "Successfully downloaded and extracted $ExpectedFile"
            Exit 0
        } else {
            # Look for any CSV file in the directory
            $csvFiles = Get-ChildItem -Path $RawDataDir -Filter *.csv -ErrorAction SilentlyContinue
            if ($csvFiles) {
                # Rename the first CSV file to expected name
                $firstCsv = $csvFiles[0]
                Rename-Item -Path $firstCsv.FullName -NewName (Join-Path $RawDataDir $ExpectedFile)
                Write-Host "Renamed $($firstCsv.Name) to $ExpectedFile"
                Exit 0
            }
        }
    } catch {
        Write-Warning "Kaggle download failed: $_"
    }
} else {
    Write-Warning "Kaggle CLI not found. Please install it following https://github.com/Kaggle/kaggle-api"
}

# If we reach here, automated download failed
Write-Host "`nAUTOMATED DOWNLOAD FAILED`n"
Write-Host "Please follow these manual steps:"
Write-Host "1. Go to https://www.kaggle.com/datasets/nicolasw/assistments-2009-2010"
Write-Host "2. Download the dataset (requires Kaggle account)"
Write-Host "3. Extract the ZIP file"
Write-Host "4. Copy 'skill_builder_data.csv' to the '$RawDataDir' directory"
Write-Host "5. Or run the synthetic fallback: python -m knowtrace.data.load --synthetic"

# Provide synthetic fallback option
Write-Host "`nAlternatively, you can generate synthetic data to test the pipeline:"
Write-Host "python -m knowtrace.data.load --synthetic"