# Creating `submission.zip` on Windows — Issues & Solution

Notes from packaging the HW8 submission on a Windows machine, where the
provided `bash zip_submission.sh` could not be run as-is.

## What the official script does

`zip_submission.sh` (Unix/bash):

```bash
rm -rf submission
mkdir submission
cp -r src setup.py requirements.txt submission/
cd submission
zip -qr ../submission.zip . -x <pycache/logs/wandb/egg-info exclusions>
```

It copies the mandatory files (`src/`, `setup.py`, `requirements.txt`) into a
`submission/` folder, excludes caches/logs, and zips them into `submission.zip`.

## Issues faced

### 1. `bash` resolves to WSL, which is not installed
Running `bash zip_submission.sh` failed with:

```
<3>WSL (9 - Relay) ERROR: CreateProcessCommon:881: execvpe(/bin/bash) failed: No such file or directory
Command exited with code 1
```

Windows ships a `bash.exe` shim (`C:\WINDOWS\system32\bash.exe`) that forwards
to WSL. With no WSL distro installed, it cannot launch the script.

### 2. Git Bash has no `zip` binary
Falling back to Git's bash (`C:\Program Files\Git\bin\bash.exe`) ran the script
but failed at the zip step:

```
zip_submission.sh: line 7: zip: command not found
```

Git for Windows does not bundle the Unix `zip` utility.

### 3. `Compress-Archive` writes backslash path separators (the real trap)
Using PowerShell's `Compress-Archive` produced a zip, but entries were stored
with Windows backslashes:

```
src\version.py
src\retriever\collator.py
```

The ZIP spec requires **forward slashes** (`/`) as the path separator. On Linux
(e.g. Gradescope's autograder) a backslash is a *legal filename character*, not
a folder separator — so the archive extracts to flat files literally named
`src\retriever\collator.py` instead of a nested `src/retriever/` tree. The
autograder then can't find the expected paths and the submission fails.

## Correct solution

Use the .NET `ZipFile` API directly so entry names can be forced to use forward
slashes. This reproduces the shell script's result in a Linux-compatible way.

```powershell
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root    = (Resolve-Path .).Path
$staging = Join-Path $root 'submission'
$zipPath = Join-Path $root 'submission.zip'

# Clean previous artifacts
Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
Remove-Item -Force $zipPath -ErrorAction SilentlyContinue

# Replicate zip_submission.sh: copy mandatory files
New-Item -ItemType Directory -Path $staging | Out-Null
Copy-Item -Recurse (Join-Path $root 'src') (Join-Path $staging 'src')
Copy-Item (Join-Path $root 'setup.py')         (Join-Path $staging 'setup.py')
Copy-Item (Join-Path $root 'requirements.txt') (Join-Path $staging 'requirements.txt')

# Strip excluded artifacts (__pycache__, logs, wandb, *.egg-info)
Get-ChildItem -Path $staging -Recurse -Directory |
    Where-Object { $_.Name -in '__pycache__','logs','wandb' -or $_.Name -like '*.egg-info' } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Build zip with forward-slash entry names
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($f in Get-ChildItem -Path $staging -Recurse -File) {
        $rel = $f.FullName.Substring($staging.Length + 1) -replace '\\','/'
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $rel) | Out-Null
    }
} finally {
    $zip.Dispose()
}
```

### Verify the entries use `/`

```powershell
$zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path .\submission.zip).Path)
"Entries: $($zip.Entries.Count)"
$zip.Entries | ForEach-Object { $_.FullName }
$zip.Dispose()
```

Expected output — 26 entries, all forward-slashed:

```
requirements.txt
setup.py
src/version.py
src/pytest_utils/decorators.py
...
src/retriever/scripts/train_msmarco.sh
```

## Alternative options

- **Run on Linux/WSL/EC2** with the Unix `zip` tool and just use the original
  `bash zip_submission.sh` — no workaround needed.
- **Install a real `zip`** on Windows (e.g. via `choco install zip` or MSYS2),
  then run the script through Git Bash.

## Takeaways

- A `submission.zip` that looks fine on Windows can still break a Linux
  autograder purely because of `\` vs `/` path separators.
- Prefer the `.NET ZipFile` approach (or a genuine `zip`/WSL) over
  `Compress-Archive` on Windows PowerShell 5.1 for cross-platform archives.
- Always verify archive entry names before uploading.
