# Windows and WSL2 setup

Run development commands inside Ubuntu in WSL2, from the repository directory. Docker Desktop should be installed with **Use the WSL 2 based engine** enabled, and Ubuntu enabled under **Settings → Resources → WSL Integration**.

Create `%UserProfile%\.wslconfig` with these settings, then run `wsl --shutdown` in PowerShell and reopen Ubuntu:

```ini
[wsl2]
memory=10GB
processors=4
```

From the Ubuntu shell, install Python 3.11, `python3.11-venv`, Docker CLI integration, kubectl, and Minikube as needed. Then run:

```bash
make setup
docker compose -f docker/docker-compose.yml up -d
minikube start --memory=5120 --cpus=3 --driver=docker
```

The current workstation agent is running in Windows PowerShell, not the requested WSL2 Ubuntu environment. Linux package installation, Docker Desktop integration, and Minikube runtime behavior must therefore be verified from WSL2.

