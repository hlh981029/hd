import os
import sys
import time
import random
import string
import configs

sys.path.append(configs.root_dir)
from flask import Flask, request, send_from_directory
from flask_cors import CORS
# import hcrm_dupl_removal as hcrm
import hcrm as hcrm
from hashlib import md5
import numpy as np
from gen_demo import gen_demo

app = Flask(__name__)
CORS(app, supports_credentials=True)
hcrm.gen_sample()

@app.route('/hdapi/get_label', methods=['POST'])
def get_label():
    if 'file' not in request.files:
        return 'no file was uploaded'
    else:
        file = request.files['file']
        file_bytes = file.read()
        filename = str(int(time.time())) + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + '.' + file.filename.split('.')[-1]
        if not os.path.exists(configs.upload_dir + filename):
            os.makedirs(configs.upload_dir + filename)
        if not os.path.exists(configs.upload_dir + filename + '/' + filename):
            file = open(configs.upload_dir + filename + '/' + filename, 'wb')
            file.write(file_bytes)
            file.close()
        result = hcrm.get_labels(filename)
        label = int(np.argmax(result))
    return {
        'filename': filename,
        'result': result,
        'label': label,
    }

@app.route('/hdapi/get_kernel_images', methods=['POST'])
def get_kernel_images():
    json_data = request.get_json()
    filename = json_data['filename']
    label = json_data['label']
    indexes = json_data['indexes']
    length = len(indexes)
    assert 0 <= length <= 4
    if length == 0:
        result = hcrm.get_kernel_images_5(filename, label)
    elif length == 1:
        result = hcrm.get_kernel_images_4(filename, label, *indexes)
    elif length == 2:
        result = hcrm.get_kernel_images_3(filename, label, *indexes)
    elif length == 3:
        result = hcrm.get_kernel_images_2(filename, label, *indexes)
    elif length == 4:
        result = hcrm.get_kernel_images_1(filename, label, *indexes)
    else:
        result = 'error'
    result_mas = result[1]
    result = result[0]
    return {
        'result': result,
        'result_mas': result_mas
    }

@app.route('/hdapi/get_demo_label', methods=['POST'])
def get_demo_label():
    json_data = request.get_json()
    filename = json_data['filename']
    result = hcrm.get_labels(filename)
    label = int(np.argmax(result))
    return {
        'result': result,
        'label': label,
    }


@app.route('/hdapi/gen_demo', methods=['POST'])
def gen_demo_image():
    json_data = request.get_json()
    filename = json_data['filename']
    label = json_data['label']
    select = json_data['indexes']
    result = hcrm.get_labels(filename)
    length = len(select)
    assert length == 4
    
    inds_5, contris_5 = hcrm.get_kernel_images_5(filename, label)
    inds_4, contris_4 = hcrm.get_kernel_images_4(filename, label, select[0])
    inds_3, contris_3 = hcrm.get_kernel_images_3(filename, label, select[0], select[1])
    inds_2, contris_2 = hcrm.get_kernel_images_2(filename, label, select[0], select[1], select[2])
    inds_1, contris_1 = hcrm.get_kernel_images_1(filename, label, select[0], select[1], select[2], select[3])
    inds = [inds_5[:4], inds_4[:4], inds_3[:4], inds_2[:4], inds_1[:4]]
    contris = [contris_5[:4], contris_4[:4], contris_3[:4], contris_2[:4], contris_1[:4]]
    res = gen_demo(filename, result, label, select, inds, contris)
    return {
        'result': res
    }

@app.route('/hdapi/image/<path:filename>')
def get_image(filename):
    return send_from_directory(configs.upload_dir, filename)

@app.route('/hdapi/pattern/<path:filename>')
def get_pattern(filename):
    return send_from_directory(configs.pattern_dir, filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
    print(dir(configs))
