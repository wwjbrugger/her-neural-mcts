import random
import shutil


def split_dataset(path):
    copy_info_file(path)
    move_data_to_testset(path)


def move_data_to_testset(path_dataset):
    p = path_dataset / "train" / "pandas"
    files = [x.name for x in p.glob("**/*") if x.is_file()]
    p_copy_data_to_testset = 1 / 4
    for file in files:
        if decision(probability_true=p_copy_data_to_testset):
            for approach in ["pandas"]:
                src_path= path_dataset / "train" / approach / file
                dst = path_dataset / "test" / approach / file
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src_path, dst)


def copy_info_file(path_dataset):
    p = path_dataset / "train"
    info_files = [x.name for x in p.glob("*") if x.is_file()]
    for info_file in info_files:
        src_path= path_dataset / "train" / info_file
        dst = path_dataset / "test" / info_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dst)


def decision(probability_true):
    return random.random() < probability_true

