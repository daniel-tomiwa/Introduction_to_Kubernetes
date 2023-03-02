# Introduction to Kubernetes


<p align="center">
  <img src="images/architecture/arch_2.png"/>
</p>

This branch in this repository contains a basic implementation of a machine learning prediction microservice using flask as the web framework. The prediction service is packaged using docker and deployed on a single node cluster while experimenting with some of the basic commands in Kubernetes.

## Pre-Requisites
1. Docker: https://www.docker.com/, https://hub.docker.com/
2. minikube (or any other kubernetes cluster): [minikube documentation](https://minikube.sigs.k8s.io/docs/start/#:~:text=Download%20and%20run%20the%20installer%20for%20the%20latest%20release.&text=Add%20the%20minikube.exe%20binary,to%20run%20PowerShell%20as%20Administrator.&text=If%20you%20used%20a%20terminal,reopen%20it%20before%20running%20minikube.)
3. virtual environment: pipenv or virtualenv (or any virtual environment you prefer)
5. kubectl

## Commands

> To install minikube (For a linux machine) Note that other machine types can follow the documentation in the link above
1. Download the minikube binary file
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
```
2. Install the downloaded binary file
```bash
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
3. Make sure you start minikube with the appropriate [driver](https://minikube.sigs.k8s.io/docs/drivers/)
```bash
minikube start --driver=docker
minikube config set driver docker
```

> To install kubectl
1. Download the kubectl binary file
```bash
curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
```
2. Make the kubectl binary file executable
```bash
chmod +x ./kubectl
```
3. Move trhe binary file to your path
```bash
sudo mv ./kubectl /usr/local/bin/kubectl
```

## Working on the Kubernetes Cluster

