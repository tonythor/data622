#!/bin/bash

# Function to build the Quarto project
build() {
    echo "Building HTML from Quarto for project${project}..."
    quarto render ./project${project}.qmd --to html
}

# Function to upload to RPubs
upload_rpubs() {
    echo "Uploading project${project} to RPubs..."
    Rscript -e "rsconnect::rpubsUpload('Project ${project}', 'project${project}.html', 'project${project}.qmd')"
}

# Ensure a project number is passed
usage() {
    echo "Usage: $0 -p <project_number> -h (build) | -r (upload to RPubs) | -a (all)"
    echo "You must specify a project number, e.g., -p 2."
    exit 1
}

# Parse options
project=""
while getopts "p:hra" opt; do
  case $opt in
    p)
      project=$OPTARG
      ;;
    h)
      [ -z "$project" ] && usage  # Check if project is empty
      build
      ;;
    r)
      [ -z "$project" ] && usage  # Check if project is empty
      upload_rpubs
      ;;
    a)
      [ -z "$project" ] && usage  # Check if project is empty
      build
      upload_rpubs
      ;;
    *)
      usage
      ;;
  esac
done

# Check if the project parameter was provided
[ -z "$project" ] && usage