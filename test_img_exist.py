import os
import pandas as pd
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def remove_img(param):
    path = param[0]
    image_tmpl = param[1]
    if os.path.exists(path):
        print(path)
        for index in range(len(os.listdir(path))):
            img_path = os.path.join(path, image_tmpl.format(int(index)))
            try:
                Image.open(img_path).convert('RGB')
            except:
                shutil.rmtree(path)
                print("remove.........")
                break

def test_img_exist(path, image_tmpl='{:03d}.jpg'):
    '''
    if img is not exist, remove the folder of img
    '''
    # csv_info = pd.read_csv(info_file)
    # csv_list = csv_info.values.tolist()

    path_list = []
    for item_id in os.listdir(path):
        if item_id[0] != ".":
            path_list.append((os.path.join(path, item_id), image_tmpl))
    with ThreadPoolExecutor(max_workers=64) as executor:
        executor.map(remove_img, path_list)

path = "/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1/frames"
# path = "/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2/frames"
image_tmpl = '{:03d}.jpg'
test_img_exist(path, image_tmpl)


