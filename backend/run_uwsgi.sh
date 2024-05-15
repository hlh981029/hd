CUDA_VISIBLE_DEVICES=1 uwsgi --socket 127.0.0.1:8003 --wsgi-file app.py --callable app --threads=1
