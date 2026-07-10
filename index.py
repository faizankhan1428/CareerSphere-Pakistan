"""
CareerSphere Pakistan - WSGI Entry Point for Vercel Serverless Deployment
Engineered by Muhammad Faizan
"""

from app import app as application

# Vercel serverless function handler
app = application

if __name__ == "__main__":
    app.run()
