# Docker - Container Platform

Docker is a platform for developing, shipping, and running applications in containers. Containers package applications with their dependencies for consistent deployment.

## Installation

### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Start service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional)
sudo usermod -aG docker $USER
```

### macOS/Windows

Download Docker Desktop from https://www.docker.com/products/docker-desktop

## Basic Commands

### Images

```bash
# List images
docker images
docker image ls

# Pull image
docker pull ubuntu:22.04
docker pull nginx:latest

# Build image
docker build -t myapp:1.0 .

# Remove image
docker rmi myapp:1.0
docker image rm myapp:1.0

# Tag image
docker tag myapp:1.0 registry.example.com/myapp:1.0

# Push image
docker push registry.example.com/myapp:1.0
```

### Containers

```bash
# Run container
docker run nginx
docker run -d nginx                    # Detached
docker run -it ubuntu bash             # Interactive
docker run --name mycontainer nginx    # Named
docker run -p 8080:80 nginx           # Port mapping
docker run -v /host:/container nginx   # Volume mount
docker run --rm nginx                  # Remove after exit

# List containers
docker ps                              # Running
docker ps -a                           # All

# Container management
docker start container_id
docker stop container_id
docker restart container_id
docker rm container_id

# Exec into container
docker exec -it container_id bash
docker exec container_id ls /

# Logs
docker logs container_id
docker logs -f container_id            # Follow
docker logs --tail 100 container_id    # Last 100 lines

# Inspect
docker inspect container_id
```

## Dockerfile

### Basic Dockerfile

```dockerfile
FROM ubuntu:22.04

# Metadata
LABEL maintainer="dev@example.com"

# Environment variables
ENV APP_HOME=/app
ENV NODE_ENV=production

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY package.json .
COPY src/ ./src/

# Install app dependencies
RUN npm install

# Expose port
EXPOSE 3000

# Run command
CMD ["npm", "start"]
```

### Multi-Stage Build

```dockerfile
# Build stage
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:18-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### Best Practices

```dockerfile
# Use specific versions
FROM node:18.19.0-alpine

# Combine RUN commands
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# Use .dockerignore
# node_modules, .git, *.log, etc.

# Non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:3000/health || exit 1
```

## Docker Compose

### docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://db:5432/myapp
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
```

### Compose Commands

```bash
# Start services
docker compose up
docker compose up -d              # Detached
docker compose up --build         # Rebuild images

# Stop services
docker compose down
docker compose down -v            # Remove volumes too

# View logs
docker compose logs
docker compose logs -f web        # Follow specific service

# Execute command
docker compose exec web bash

# Scale services
docker compose up --scale web=3
```

## Networking

```bash
# List networks
docker network ls

# Create network
docker network create mynetwork

# Run container on network
docker run --network mynetwork nginx

# Connect container to network
docker network connect mynetwork container_id

# Inspect network
docker network inspect mynetwork
```

## Volumes

```bash
# List volumes
docker volume ls

# Create volume
docker volume create myvolume

# Use volume
docker run -v myvolume:/data nginx

# Bind mount
docker run -v /host/path:/container/path nginx

# Remove volume
docker volume rm myvolume

# Prune unused volumes
docker volume prune
```

## Registry

```bash
# Login to registry
docker login
docker login registry.example.com

# Push to registry
docker push registry.example.com/myapp:1.0

# Pull from registry
docker pull registry.example.com/myapp:1.0

# Run local registry
docker run -d -p 5000:5000 registry:2
```

## Resource Limits

```bash
# Memory limit
docker run -m 512m nginx

# CPU limit
docker run --cpus=2 nginx
docker run --cpu-shares=512 nginx

# Combined
docker run -m 1g --cpus=2 myapp
```

## Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune
docker image prune -a              # All unused

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune
docker system prune -a --volumes   # Nuclear option

# Disk usage
docker system df
```

## Security

```bash
# Run as non-root
docker run --user 1000:1000 nginx

# Read-only filesystem
docker run --read-only nginx

# No new privileges
docker run --security-opt=no-new-privileges nginx

# Scan for vulnerabilities
docker scan myimage:latest
```

## Documentation

- Official: https://docs.docker.com/
- Dockerfile Reference: https://docs.docker.com/engine/reference/builder/
- Compose: https://docs.docker.com/compose/
- Docker Hub: https://hub.docker.com/
