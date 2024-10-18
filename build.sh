#!/bin/bash

# Function to build the Quarto project
build() {
    echo "Building HTML from Quarto..."
    quarto render ./project1.qmd --to html
}

# Function to upload to RPubs
upload_rpubs() {
    echo "Uploading to RPubs..."
    Rscript -e "rsconnect::rpubsUpload('622 Project 1', 'project1.html', 'project1.qmd')"
}

# Check the argument passed
while getopts "hra" opt; do
  case $opt in
    h)
      build
      ;;
    r)
      upload_rpubs
      # requires the "rsconnect" folder from previous project!
      # returns the following, go to the continue url and claim the page.
      # [1] "https://api.rpubs.com/api/v1/document/1229195/1b9ef99cdcaf419ba3c51a101056d0ae"
      # $continueUrl
      # [1] "http://rpubs.com/publish/claim/1229195/a4db2923e69e43fb82a62a12c795d9e3"
      ;;
    a)
      build
      upload_rpubs
      ;;
    *)
      echo "Invalid option: -$opt"
      echo "Usage: $0 -h (build to html) | -r (rpubs upload) | -a (all features)"
      exit 1
      ;;
  esac
done


