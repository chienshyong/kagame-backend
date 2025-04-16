# Kagame Backend
A backend service for Kagame, a fashion-focused application that helps users reflect their style.

## Table of Contents
- [Kagame Backend](#kagame-backend)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [1. Google Cloud SDK Setup](#1-google-cloud-sdk-setup)
    - [2. Python Environment Setup](#2-python-environment-setup)
  - [Configuration](#configuration)
  - [Development](#development)
    - [Running the Server](#running-the-server)
    - [Project Structure](#project-structure)
  - [API Documentation](#api-documentation)
  - [Architecture](#architecture)
  - [Deployment](#deployment)
    - [RunPod Deployment](#runpod-deployment)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [Security](#security)

## Overview
Kagame Backend provides the server-side functionality for the Kagame application, handling:
- User authentication and authorization
- Image processing and storage
- Database operations
- API endpoints for client applications

## Prerequisites
- Python 3.8 or higher
- Google Cloud SDK
- Docker (for deployment)
- MongoDB (for database operations)

## Installation

### 1. Google Cloud SDK Setup
Google Cloud Storage uses Application Default Credentials, not API keys. Follow these steps if you haven't installed Google Cloud SDK:

1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
2. Initialize and authenticate:
   ```bash
   gcloud init
   gcloud auth application-default login
   ```
3. Login with Kagame's gmail account when prompted
4. Verify credentials are saved (you should see a message like):
   ```
   Credentials saved to file: [some\local\filepath\gcloud\application_default_credentials.json]
   ```

### 2. Python Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/macOS

# Install dependencies
pip install -r "requirements.txt"
```

## Configuration
1. Create a `secretstuff` directory:
   ```bash
   cp -r secretstuff-example secretstuff
   ```
2. Populate secretstuff with the relevant files from the team chat

## Development

### Running the Server
Development mode (with auto-reload):
```bash
uvicorn main:app --reload
```

Production mode:
```bash
uvicorn main:app
```

### Project Structure
- `/routes`: API endpoints organized by functionality
  - Use `auth.py` for endpoint protection and user identification
- `kagameDB.py`: Database connection management
- `services/image.py`: Image handling service
  - Currently stores images as binary data in MongoDB
  - Designed for easy migration to file system storage

## API Documentation
API documentation is available at `/docs` when running the server (Swagger UI).

Key endpoints:
- Authentication endpoints (see `routes/auth.py`)
- Image processing endpoints
- User management endpoints

## Architecture
The backend follows a modular architecture:
- FastAPI for the web framework
- MongoDB for data storage
- Google Cloud Storage for file storage
- JWT for authentication

## Deployment
### RunPod Deployment
1. Build Docker image:
   ```bash
   docker build -t kagame-backend .
   docker run -d -p 80:80 kagame-backend
   ```

2. Push to Docker Hub:
   ```bash
   docker login -u kagameteam
   docker tag kagame-backend kagameteam/kagame-backend:latest
   docker push kagameteam/kagame-backend:latest
   ```

Use `kagameteam/kagame-backend` as the image on RunPod container.

## Troubleshooting
Common issues and solutions:
1. Google Cloud authentication issues
   - Verify credentials are properly set up
   - Check if `gcloud auth application-default login` completed successfully
2. Database connection issues
   - Verify MongoDB is running
   - Check connection string in configuration
3. Image processing errors
   - Ensure all required dependencies are installed
   - Verify Google Cloud Storage permissions

## Contributing
1. Create a new branch for your feature
2. Follow the existing code style
3. Add tests for new functionality
4. Submit a pull request with a clear description of changes

## Security
- Never commit sensitive information or API keys
- Always use environment variables for secrets
- Keep dependencies updated
- Follow security best practices for authentication