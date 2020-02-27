import os


### define following constants ###

# 1. where to create train symlinks
TRAIN_PATH = '/home/user/project/point_patches/train/'
# 2. where to create test symlinks
TEST_PATH = '/home/user/project/point_patches/test/'
# 3. where actual data is
DATA_PATH = '/home/user/project/data/'
# 4. where train.txt/test.txt are
TXT_FILES_PATH = '/home/user/project/point_patches/'

if __name__ == '__main__':
    train_list = []
    with open(os.path.join(TXT_FILES_PATH, 'train.txt'), 'r') as f:
        for item in f.readlines():
            train_list.append(item.strip())
    test_list = []
    with open(os.path.join(TXT_FILES_PATH, 'test.txt'), 'r') as f:
        for item in f.readlines():
            test_list.append(item.strip())

    for item in train_list:
        name = item.split('/')[-1]
        src = os.path.join(DATA_PATH, name)
        dst = os.path.join(TRAIN_PATH, name)
        os.symlink(src, dst)
    for item in test_list:
        name = item.split('/')[-1]
        src = os.path.join(DATA_PATH, name)
        dst = os.path.join(TEST_PATH, name)
        os.symlink(src, dst)