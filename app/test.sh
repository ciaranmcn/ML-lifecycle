#!/bin/bash

export WORKFLOW_ID=$(uuidgen)
echo "Workflow: $WORKFLOW_ID"

curl -X POST "http://localhost:8000/start/$WORKFLOW_ID" 


curl -X POST "http://localhost:8000/send/$WORKFLOW_ID" \
  -H "Content-Type: application/json" \
  -d '"This is my feedback"'

curl "http://localhost:8000/result/$WORKFLOW_ID" 

