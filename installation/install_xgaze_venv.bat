C:\Users\oslab\AppData\Local\Programs\Python\Python310\python.exe -m venv ../third_party/xgaze_3cams/env
call ../third_party/xgaze_3cams/env/Scripts/activate
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r ../third_party/xgaze_3cams/requirements.txt

