#!/bin/bash

git add eagle
git add *.py
git add pip.txt
git add .gitignore
git add git.sh

git commit -m "$(date +'%Y-%m-%d')"
git push
