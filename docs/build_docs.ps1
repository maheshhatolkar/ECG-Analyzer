param(
    [string]$python = '.\.venv\Scripts\python.exe'
)

Write-Host "Installing Sphinx requirements and building docs..."
$env:PYTHON = $python
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

# Build docs
mkdir -Force docs\_build | Out-Null
& $python -m sphinx -b html docs docs\_build\html
Write-Host "Docs built at docs\_build\html"
