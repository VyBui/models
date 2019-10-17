import os.path

path = '/home/vybt/dataset_cropped/train/bodyparts'

dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

for folder in os.listdir(path):
    path_folder = os.path.join(path, folder)
    path_new_folder = path_folder.split(".")[0]
    if os.path.isdir(path_folder):
        os.rename(path_folder, path_new_folder)
        print("change from: " + path_folder + " => " + path_new_folder)


