#!/bin/bash
# Git alias setup for easy pushing
# Run this once to set up git aliases for automatic pushing

echo "Setting up git aliases for easy GitHub updates..."

# Add alias for quick push
git config --local alias.quick-push '!f() { git add . && git commit -m "${1:-Auto-update}" && git pull origin main --rebase && git push origin main; }; f'

# Add alias for status and push
git config --local alias.publish '!f() { git status && git add . && git commit -m "${1:-Auto-update}" && git pull origin main --rebase && git push origin main; }; f'

echo "Git aliases set up successfully!"
echo ""
echo "Usage:"
echo "  git quick-push \"Your commit message\""
echo "  git publish \"Your commit message\""

