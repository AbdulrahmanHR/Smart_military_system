# Deployment Fix Guide

## Issues Fixed

### 1. Bad Gateway Resolution
The bad gateway error was likely caused by:
- Missing environment variables (especially `GOOGLE_API_KEY`)
- CORS configuration issues
- Health check failures

### 2. Key Changes Made

#### A. Environment Configuration
- Created `.env.example` with proper configuration
- **YOU NEED TO CREATE `.env` FILE** with your actual `GOOGLE_API_KEY`

#### B. CORS Configuration
- Updated CORS to use environment-configurable origins
- Default includes your domain: `https://smart.ibrahimshaikh.com.tr`
- Supports both production and development

#### C. Enhanced Health Checks
- Improved health endpoint with diagnostics
- Better error reporting for troubleshooting
- Extended Docker health check timeout

#### D. Better Error Handling
- Enhanced frontend serving with proper error messages
- Improved logging for debugging

## Steps to Deploy

### 1. Create Environment File
```bash
cp .env.example .env
```

Then edit `.env` and add your actual Google API key:
```
GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 2. Build and Deploy
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 3. Check Health
Visit: `https://smart.ibrahimshaikh.com.tr/health`

This should show:
```json
{
    "status": "healthy",
    "message": "Military Training Chatbot - Status: healthy. Details: {...}"
}
```

### 4. Check Logs
```bash
docker-compose logs -f military-chatbot
```

## Common Issues & Solutions

### Issue: Still Getting Bad Gateway
**Solution:** Check your reverse proxy (nginx/apache) configuration:
- Ensure it's forwarding to port 8000
- Check proxy_pass or ProxyPass directive
- Verify SSL certificates

### Issue: Health Check Shows "degraded"
**Solution:** 
- Verify `GOOGLE_API_KEY` is set correctly in `.env`
- Check if the API key is valid at https://aistudio.google.com/
- Ensure directories exist: `documents/`, `database/`, `uploads/`

### Issue: CORS Errors
**Solution:**
- Add your domain to `CORS_ORIGINS` in `.env`:
```
CORS_ORIGINS=https://smart.ibrahimshaikh.com.tr,https://yourdomain.com
```

### Issue: Frontend Not Loading
**Solution:**
- Check that `frontend/index.html` exists
- Verify Docker volumes are properly mounted
- Check logs for file system errors

## Verification Commands

```bash
# Check if container is running
docker-compose ps

# Check container logs
docker-compose logs military-chatbot

# Check health endpoint
curl https://smart.ibrahimshaikh.com.tr/health

# Test API endpoint
curl -X POST https://smart.ibrahimshaikh.com.tr/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello", "language": "en"}'
```

## No Qdrant Found

✅ **Confirmed:** Your system is already using ChromaDB, not Qdrant. No migration needed.

## Next Steps

1. Create the `.env` file with your Google API key
2. Redeploy using the commands above
3. Test the health endpoint
4. If issues persist, check your server's reverse proxy configuration
