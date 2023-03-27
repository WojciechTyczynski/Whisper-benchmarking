# Setup environment
Modules: \
module load cuda/11.3 \
module load ffmpeg \
module load python3/3.9.11 \


install torch like this: \
`pip3 install torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu113`

install other dependencies: \
`pip3 install -r requirements.txt`\
`pip install git+https://github.com/openai/whisper.git`

# Usefull commands 
start notebook on remote server
`jupyter notebook --no-browser --port=8888 --ip=$HOSTNAME`

on local machine
`ssh s183954@login1.hpc.dtu.dk -g -L8888:n-62-12-19:8888 -N`

Watch ram usage of job in queue 
`bnvtop <jobid>`