#!/bin/bash

# Configuration : On demande la ersion (ex: v3)
VERSION=$1
if [ -z "$VERSION" ]; then
  echo "❌ Usage: ./deploy.sh v3"
  exit 1
fi

IMAGE_NAME="maintenance-app:$VERSION"

echo "🚀 Starting deployment for version $VERSION..."

# 1. Building the image
echo "📦 1/3 Building Docker image..."
sudo docker build -t $IMAGE_NAME .

# 2. Loading into Minikube
echo "🚚 2/3 Transferring image to Minikube..."
minikube image load $IMAGE_NAME

# 3. YAML Update and Deployment
echo "☸️ 3/3 Updating Kubernetes..."
# Replacing the old version with the new one in the YAML file
sed -i "s/maintenance-app:.*/maintenance-app:$VERSION/g" deployment.yaml
kubectl apply -f deployment.yaml

echo "✅ Finished! Your AI is up to date in version $VERSION."
echo "Checking pods:"
kubectl get pods