#!/usr/bin/bash

#Create the file in the duration-prediction container
kubectl exec -it $PODNAME -c $CONTAINER_NAME -- touch /$mountPath/logfile.txt

#Read the file from the sidecar container
kubectl exec -it $PODNAME -c $CONTAINER_NAME -- ls -l /$mountPath