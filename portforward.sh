#!/bin/bash

osascript <<EOF
tell application "Terminal"
    activate
    do script "brev port-forward cosmos-videos-gpu -p 8501:8501"
end tell
EOF

#make it executable: chmod +x portforward.sh
#run it: ./portforward.sh

#This will: Open a new terminal window and automatically run your port-forward command