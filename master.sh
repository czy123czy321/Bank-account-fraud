#! /usr/bin/bash

jupyter nbconvert --execute /Users/zoe/Documents/Bank-account-fraud/code/eda.ipynb --to html --output-dir /Users/zoe/Documents/Bank-account-fraud/output
 	
wait

cat 'EDA exported into html'

jupyter nbconvert --execute /Users/zoe/Documents/Bank-account-fraud/code/eda.ipynb --to script
python /Users/zoe/Documents/Bank-account-fraud/code/eda.py

wait

cat 'Done running the eda.py'
