#!/bin/bash

#env.sh is not pushed to the repo (.gitignore)
source env.sh
bash setup.sh

echo " Launching app..."
python 3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
#then run "bash run.sh"