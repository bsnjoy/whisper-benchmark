# whisper-benchmark
Test how fast is your GPU running Whisper

## Install
```
sudo apt update
sudo apt install -y git ffmpeg python3-pip python3.11-venv

git clone https://github.com/bsnjoy/whisper-benchmark.git benchmark
cd benchmark
mkdir venv
python3 -m venv venv/
. venv/bin/activate
pip install -r requirements.txt
cp config.py.sample config.py
# Run
python3 benchmark.py
```
