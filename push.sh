#!/bin/bash
# Quick git push script for LOCAL_LLM project
# Usage: ./push.sh "commit message"

MSG="${1:-update}"
TOKEN=$(cat token.txt 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "❌ token.txt not found or empty"
    exit 1
fi

git remote set-url origin "https://mendeltem:${TOKEN}@github.com/mendeltem/local_llm_agents.py.git"

git add .
git status
echo ""
read -p "Push with message '$MSG'? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    git commit -m "$MSG"
    git push
    echo "✅ Pushed!"
else
    echo "❌ Aborted"
    git reset HEAD . > /dev/null 2>&1
fi
