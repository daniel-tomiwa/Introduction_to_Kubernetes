#!/usr/bin/bash

echo 'Creating the healthy file for the readiness probe'

for name in $(kubectl get pods | grep '.*2/2.*' | cut -d ' ' -f 1);
do
    echo $name
    kubectl exec $name -c simpleapp -- touch /tmp/healthy
done

echo 'healthy files created in all inactive pods'