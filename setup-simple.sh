#!/bin/bash
set -e

echo "🚀 Setting up AI Lead Scoring Engine (Simple Version)..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    echo "   You can start Docker Desktop from Applications or run:"
    echo "   open /Applications/Docker.app"
    exit 1
fi

echo "✅ Docker is running"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "   Please edit .env file with your configuration"
fi

# Create models directory
mkdir -p models

# Build and start services
echo "🏗️  Building and starting services..."
docker compose -f deployment/docker-compose.simple.yml build
docker compose -f deployment/docker-compose.simple.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker compose -f deployment/docker-compose.simple.yml ps

echo "🌐 Services should be available at:"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - Health Check: http://localhost:8000/health"

# Test the API
echo "🧪 Testing API health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy!"
else
    echo "⚠️  API is not ready yet. Check logs with:"
    echo "   docker compose -f deployment/docker-compose.simple.yml logs lead_scoring_api"
fi

echo "🎉 Setup complete!"
echo ""
echo "To view logs: docker compose -f deployment/docker-compose.simple.yml logs -f"
echo "To stop: docker compose -f deployment/docker-compose.simple.yml down"
echo "To restart: docker compose -f deployment/docker-compose.simple.yml restart"

# Test the API with sample data
echo ""
echo "🧪 Testing API with sample data..."
curl -X POST "http://localhost:8000/score-lead" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "lead_id": "test_lead_001",
         "property_listing_views": 10,
         "emails_opened": 5,
         "emails_sent": 8,
         "annual_income": 75000,
         "age": 32,
         "credit_score": 720,
         "employment_type": "Full-time"
       }
     }' | python -m json.tool || echo "API test failed - check logs"
