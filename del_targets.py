import os
import shutil
from PIL import Image

def delete_folders_without_enough_png(root_path):
    for dirpath, _, filenames in os.walk(root_path, topdown=False):
        if dirpath == root_path:
            continue
        
        png_files = [file for file in filenames if file.lower().endswith('.png')]
        
        if len(png_files) != 32:
            print(f"{dirpath} rejected, it contains {len(png_files)} PNG files")
            shutil.move(dirpath, os.path.join('train', 'rejectedPath'))
        else:
            for file in png_files:
                try:
                    v_image = Image.open(os.path.join(dirpath, file))
                    v_image.verify()
                except:
                    print(f'{dirpath} rejected. corrupted file {os.path.join(dirpath, file)} ')
                    try:
                        shutil.move(dirpath, os.path.join('train', 'rejectedPath'))
                    except:
                        shutil.rmtree(dirpath)
                    break
        
root_path = "train/target"
delete_folders_without_enough_png(root_path)