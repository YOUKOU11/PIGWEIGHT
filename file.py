import os
import shutil

train_path = 'D:\\Python项目\\classification-master\\data\my_dataset\\train'
train_out = 'D:\\Python项目\\classification-master\\data\my_dataset\\train.txt'
val_path = 'D:\\Python项目\\classification-master\\data\\my_dataset\\val'
val_out = 'D:\\Python项目\\classification-master\\data\\my_dataset\\val.txt'
test_path = 'D:\\Python项目\\classification-master\\data\\my_dataset\\test'
test_out = 'D:\\Python项目\\classification-master\\data\\my_dataset\\test.txt'

data_train_out = 'D:\\Python项目\\classification-master\\data\\my_dataset\\train_filelist'
data_val_out = 'D:\\Python项目\\classification-master\\data\\my_dataset\\val_filelist'
data_test_out = 'D:\\Python项目\\classification-master\\data\\my_dataset\\test_filelist'


def get_filelist(input_path, out_path):
    with open(out_path, 'w') as f:
        for dir_path, dir_names, file_names in os.walk(input_path):
            if dir_path != input_path:
                label = dir_path.split('\\')[-1]
                print(label)

            for file_name in file_names:
                f.write(file_name + ',' + str(label) + '\n')


def move_imgs(input_path, out_path):
    for dir_path, dir_names, file_names in os.walk(input_path):
        for file_name in file_names:
            source_path = os.path.join(dir_path, file_name)
            shutil.copyfile(source_path, os.path.join(out_path, file_name))


get_filelist(train_path, train_out)
get_filelist(val_path, val_out)
get_filelist(test_path, test_out)

move_imgs(train_path, data_train_out)
move_imgs(val_path, data_val_out)
move_imgs(test_path, data_test_out)
