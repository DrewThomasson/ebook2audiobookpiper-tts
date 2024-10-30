## To run this locally for local free workflow testing: ⬇️

## Install `act` via Software Package Managers

`act` is available across multiple repositories. Use the appropriate command for your OS:

### Arch (Linux)  
```bash
sudo pacman -S act
```

### Homebrew (Linux, macOS)  
```bash
brew install act
```

### Chocolatey (Windows)  
```bash
choco install act-cli
```

### COPR (Fedora-based Linux)  
```bash
sudo dnf copr enable denysvitali/act && sudo dnf install act
```

### GitHub CLI (All Platforms)  
```bash
gh extension install nektos/gh-act
```

### Nix/NixOS (Linux, macOS)  
```bash
nix-env -iA nixpkgs.act
```

### MacPorts (macOS)  
```bash
sudo port install act
```

### Scoop (Windows)  
```bash
scoop install act
```

### Winget (Windows)  
```bash
winget install nektos.act
```

---

### Run the Act Command for Testing Locally  
```bash
act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:full-latest --container-architecture linux/amd64 --verbose
``` 

With this, you can seamlessly replicate your GitHub Actions locally across platforms.
