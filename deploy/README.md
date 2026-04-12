# Deployment Guide

## Local Mac (current)
./run.sh   # starts Streamlit on port 8501

## VPS / Cloud Deployment

### Prerequisites
- Ubuntu 22.04 VPS (DigitalOcean, Linode, AWS EC2 t3.small)
- Python 3.11+, pip, git

### Setup
```bash
git clone <your-repo> ~/projects/quant-platform
cd ~/projects/quant-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create log directory
sudo mkdir -p /var/log/quant-platform
sudo chown $USER /var/log/quant-platform

# Install supervisord
pip install supervisor

# Configure environment
cp .env.example .env   # fill in API keys

# Start with supervisord
supervisord -c deploy/supervisord.conf
supervisorctl status
```

### Useful Commands
```bash
supervisorctl status           # check process status
supervisorctl restart all      # restart everything
supervisorctl tail -f quant-streamlit   # follow logs
```

### Nginx Reverse Proxy (optional)
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```
