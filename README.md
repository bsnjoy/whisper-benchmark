# whisper-benchmark
Test how fast is your GPU running Whisper

## Install
```
sudo apt update
sudo apt install -y git
sudo apt install -y python3-pip
sudo apt install -y ffmpeg

pip install git+https://github.com/openai/whisper.git
git clone https://github.com/bsnjoy/whisper-benchmark.git
cd whisper-benchmark
pip install -r requirements.txt
cp config.py.sample config.py
# Run
python3 benchmark.py
```
