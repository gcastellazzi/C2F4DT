#!/bin/bash
# Quick branch status check for GitHub repo
# Usage: ./check_branch.sh nome_branch

BRANCH=$1

if [ -z "$BRANCH" ]; then
  echo "❌ Please provide a branch name. Example:"
  echo "   ./check_branch.sh my_feature"
  exit 1
fi

echo "🔄 Fetching latest updates from origin..."
git fetch origin

echo "📍 Switching to branch: $BRANCH"
git checkout $BRANCH

echo "⬇️ Pulling latest commits..."
git pull origin $BRANCH

echo "✅ Checking status relative to origin/$BRANCH..."

# Check if branch is up-to-date
LOCAL=$(git rev-parse $BRANCH)
REMOTE=$(git rev-parse origin/$BRANCH)

if [ "$LOCAL" = "$REMOTE" ]; then
  echo "✔️ Branch '$BRANCH' is up to date with origin/$BRANCH"
else
  echo "⚠️ Branch '$BRANCH' differs from origin/$BRANCH"
  echo "Local HEAD:  $LOCAL"
  echo "Remote HEAD: $REMOTE"
  echo
  echo "👉 Differences (local ahead):"
  git log origin/$BRANCH..HEAD --oneline
  echo
  echo "👉 Differences (remote ahead):"
  git log HEAD..origin/$BRANCH --oneline
fi
