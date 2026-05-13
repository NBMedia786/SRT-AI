# Hostinger Deployment Guide - SRT-AI

Complete guide for deploying the SRT-AI subtitle generator to Hostinger hosting.

## Prerequisites

- Hostinger VPS or Business hosting plan (Node.js support required)
- SSH access to your server
- Domain name configured
- Google Gemini API key

## Step 1: Server Setup

### 1.1 Connect to Your Server

```bash
ssh username@your-server-ip
```

### 1.2 Install Node.js 18+

```bash
# Update package manager
sudo apt update

# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version  # Should show v18.x or higher
npm --version
```

### 1.3 Install FFmpeg

```bash
# Install FFmpeg for audio/video processing
sudo apt install -y ffmpeg

# Verify installation
ffmpeg -version
```

### 1.4 Install PM2 Process Manager

```bash
# Install PM2 globally
sudo npm install -g pm2

# Verify installation
pm2 --version
```

## Step 2: Deploy Application

### 2.1 Upload Your Code

Option A: Using Git (Recommended)
```bash
cd /var/www
git clone https://your-repository-url.git srt-ai
cd srt-ai
```

Option B: Using FTP/SFTP
- Upload all files to `/var/www/srt-ai` directory
- Ensure all files have correct permissions

### 2.2 Install Dependencies

```bash
cd /var/www/srt-ai
npm install --production
```

### 2.3 Configure Environment Variables

```bash
# Create .env file
nano .env
```

Add the following (replace with your actual values):
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
PORT=3000
NODE_ENV=production
GEMINI_MODEL=gemini-2.5-pro
REQUEST_TIMEOUT_MS=1800000
MAX_FILE_SIZE_MB=2048
RATE_LIMIT_REQUESTS=10
LOG_LEVEL=info
```

Save and exit (Ctrl+X, then Y, then Enter)

### 2.4 Set File Permissions

```bash
# Set ownership
sudo chown -R www-data:www-data /var/www/srt-ai

# Set permissions
sudo chmod -R 755 /var/www/srt-ai
```

## Step 3: Configure PM2

### 3.1 Create PM2 Ecosystem File

```bash
cd /var/www/srt-ai
nano ecosystem.config.js
```

Add the following:
```javascript
module.exports = {
  apps: [{
    name: 'srt-ai',
    script: './server.js',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true,
    max_memory_restart: '1G',
    autorestart: true,
    watch: false
  }]
};
```

### 3.2 Create Logs Directory

```bash
mkdir -p logs
```

### 3.3 Start Application with PM2

```bash
# Start the application
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Setup PM2 to start on system boot
pm2 startup
# Follow the command it outputs
```

### 3.4 Verify Application is Running

```bash
# Check PM2 status
pm2 status

# View logs
pm2 logs srt-ai

# Check if app is responding
curl http://localhost:3000/api/health
```

## Step 4: Configure Nginx Reverse Proxy

### 4.1 Install Nginx (if not installed)

```bash
sudo apt install -y nginx
```

### 4.2 Create Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/srt-ai
```

Add the following configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    # Increase timeouts for long-running requests
    proxy_connect_timeout 1800s;
    proxy_send_timeout 1800s;
    proxy_read_timeout 1800s;
    send_timeout 1800s;

    # Increase max body size for large file uploads
    client_max_body_size 2G;
    client_body_timeout 1800s;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Disable buffering for SSE (Server-Sent Events)
        proxy_buffering off;
        proxy_cache off;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;
}
```

### 4.3 Enable Site and Restart Nginx

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/srt-ai /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

## Step 5: Setup SSL/HTTPS (Recommended)

### 5.1 Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 5.2 Obtain SSL Certificate

```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Follow the prompts to:
- Enter your email
- Agree to terms
- Choose to redirect HTTP to HTTPS (recommended)

### 5.3 Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Certbot auto-renewal is set up automatically via cron
```

## Step 6: Monitoring & Maintenance

### 6.1 Monitor Application

```bash
# View real-time logs
pm2 logs srt-ai

# Monitor resources
pm2 monit

# View detailed info
pm2 info srt-ai
```

### 6.2 Restart Application

```bash
# Restart app
pm2 restart srt-ai

# Reload app (zero-downtime)
pm2 reload srt-ai
```

### 6.3 Update Application

```bash
cd /var/www/srt-ai

# Pull latest code (if using Git)
git pull

# Install dependencies
npm install --production

# Restart application
pm2 restart srt-ai
```

### 6.4 View Application Logs

```bash
# PM2 logs
pm2 logs srt-ai --lines 100

# Application log files
tail -f /var/www/srt-ai/logs/combined.log
```

## Troubleshooting

### Issue: Application won't start

**Check logs:**
```bash
pm2 logs srt-ai --err
```

**Common fixes:**
- Verify `.env` file exists and has correct API key
- Check Node.js version: `node --version` (must be 18+)
- Verify FFmpeg is installed: `ffmpeg -version`
- Check file permissions: `ls -la /var/www/srt-ai`

### Issue: "GEMINI_API_KEY is missing"

**Fix:**
```bash
cd /var/www/srt-ai
nano .env
# Add: GEMINI_API_KEY=your_key_here
pm2 restart srt-ai
```

### Issue: File upload fails

**Check Nginx configuration:**
```bash
sudo nano /etc/nginx/sites-available/srt-ai
# Ensure client_max_body_size is set to 2G or higher
sudo nginx -t
sudo systemctl restart nginx
```

### Issue: Processing timeout

**Increase timeouts in Nginx:**
```nginx
proxy_read_timeout 3600s;  # 1 hour
client_body_timeout 3600s;
```

**Increase timeout in .env:**
```env
REQUEST_TIMEOUT_MS=3600000  # 1 hour
```

### Issue: High memory usage

**Restart application:**
```bash
pm2 restart srt-ai
```

**Monitor memory:**
```bash
pm2 monit
```

### Issue: Cannot access via domain

**Check Nginx status:**
```bash
sudo systemctl status nginx
```

**Check DNS settings:**
- Ensure domain A record points to server IP
- Wait for DNS propagation (up to 48 hours)

**Check firewall:**
```bash
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

## Performance Optimization

### 1. Enable Nginx Caching

Add to Nginx config:
```nginx
location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

### 2. Increase PM2 Instances (if you have multiple CPU cores)

```bash
pm2 scale srt-ai 2  # Run 2 instances
```

### 3. Monitor Disk Space

```bash
# Check disk usage
df -h

# Clean up old logs
pm2 flush
```

## Security Best Practices

1. **Keep API key secure** - Never commit `.env` to Git
2. **Use HTTPS** - Always use SSL certificates
3. **Regular updates** - Keep Node.js, npm, and dependencies updated
4. **Firewall** - Only allow necessary ports (80, 443, 22)
5. **Rate limiting** - Already configured in the app
6. **Monitor logs** - Regularly check for suspicious activity

## Health Check

Test your deployment:

```bash
# Check health endpoint
curl https://your-domain.com/api/health

# Should return:
# {"status":"online","system":"Gemini 2.x Architecture",...}
```

## Support

If you encounter issues:
1. Check application logs: `pm2 logs srt-ai`
2. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
3. Verify all environment variables are set correctly
4. Ensure FFmpeg is installed and accessible
5. Check Gemini API quota and status

## Quick Reference Commands

```bash
# Start application
pm2 start ecosystem.config.js

# Stop application
pm2 stop srt-ai

# Restart application
pm2 restart srt-ai

# View logs
pm2 logs srt-ai

# Monitor resources
pm2 monit

# Check status
pm2 status

# Restart Nginx
sudo systemctl restart nginx

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
```
