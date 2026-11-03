# WSL2 development setup

Run project commands from Ubuntu in WSL2, with the repository under the Linux home directory (for example, `~/fraud-platform`), not under `/mnt/c`. Install Python 3.11 through uv so the system Python version does not affect the project environment.

## Windows setup

Install Docker Desktop and enable **Use the WSL 2 based engine**. Under **Settings → Resources → WSL Integration**, enable the Ubuntu distribution. Create `%UserProfile%\.wslconfig` in Windows with suitable resource limits, for example:

```ini
[wsl2]
memory=10GB
processors=4
```

Apply changes from PowerShell with `wsl --shutdown`, then reopen Ubuntu. Adjust memory and processor counts to the host machine.

## Ubuntu setup

Install uv using its official installation method, then in the Ubuntu terminal run:

```bash
cd ~/fraud-platform
uv python install 3.11
make setup
.venv/bin/python --version
```

`make setup` creates `.venv` with Python 3.11 and installs the pinned requirements. The Makefile uses uv's `unsafe-best-match` index strategy so packages such as `requests` remain available alongside the PyTorch CPU wheel index.

Check Docker Desktop integration:

```bash
docker version
docker compose version
```

If the Docker socket reports a group permission error, check `id` and the socket's group. Add the current WSL user to the `docker` group only if that group exists and is required by the installed integration:

```bash
sudo usermod -aG docker "$USER"
```

Close all Ubuntu shells and reopen the distribution for group membership to take effect. Docker Desktop WSL integration normally manages the socket; do not install a second Docker daemon inside Ubuntu unless intentionally configuring one.

## Optional local services

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Minikube is optional for the current repository because Kubernetes manifests have not been added. If installed separately, start it only after Docker works in WSL2:

```bash
minikube start --memory=5120 --cpus=3 --driver=docker
```
