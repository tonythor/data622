#!/bin/bash

mkdir -p old_versions  # Create the output directory if it doesn't exist

# Get the last 20 commits affecting the file
commits=$(git log --pretty=format:%h -n 20 -- ./project1.qmd)

# Loop over each commit and save the version
for commit in $commits; do
  git show "$commit:./project1.qmd" > old_versions/"project1_$commit.qmd"
done

echo "Downloaded the last 20 versions to the 'old_versions' directory."
