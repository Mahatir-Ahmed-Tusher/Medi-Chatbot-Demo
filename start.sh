#!/bin/bash
echo "🔄 Starting server on port $PORT"
exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 4 app:app
