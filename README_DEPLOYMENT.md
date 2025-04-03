# Deployment Guide for Product Similarity Microservice

This document outlines the steps to build, run, and deploy the Product Similarity FastAPI microservice using Docker and Kubernetes.

---

## Docker Instructions

### Build Docker Image

```bash
docker build -t product-similarity-app .
```

### Run the Container Locally

```bash
docker run -p 8000:8000 product-similarity-app
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) to test the FastAPI Swagger UI.

---

## Docker Hub Deployment

### 1. Tag Your Image

```bash
docker tag product-similarity-app your_dockerhub_username/product-similarity-app
```

### 2. Push to Docker Hub

```bash
docker push your_dockerhub_username/product-similarity-app
```

Make sure you're logged in with:

```bash
docker login
```

---

## Kubernetes Deployment

### 1. Create a Kubernetes Deployment File (`deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-similarity
spec:
  replicas: 1
  selector:
    matchLabels:
      app: product-similarity
  template:
    metadata:
      labels:
        app: product-similarity
    spec:
      containers:
      - name: product-similarity
        image: your_dockerhub_username/product-similarity-app
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: product-similarity-service
spec:
  selector:
    app: product-similarity
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: NodePort
```

### 2. Apply the Configuration

```bash
kubectl apply -f deployment.yaml
```

### 3. Access the Microservice

Use `minikube service` or `kubectl get svc` to find the NodePort exposed:

```bash
minikube service product-similarity-service
```

Or:

```bash
kubectl get svc
```

Then open the endpoint in your browser:
```
http://<node_ip>:<node_port>/find_similar_products?product_id=<id>&num_similar=5
```

---

## Cleanup Resources

```bash
kubectl delete -f deployment.yaml
```

---

You’ve successfully containerized and deployed your FastAPI app using Kubernetes. 