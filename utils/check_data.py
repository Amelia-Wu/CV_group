
import os
"""
Replace the following path with your own path
"""
train_left = "../dataset/train/left"
train_right = "../dataset/train/right"
test_left = "../dataset/test/left"
test_right = "../dataset/test/right"


def find_duplicate_files(path1, path2):
    """
    check if there are duplicate files in two directories
    """
    files_in_path1 = set(os.listdir(path1))
    files_in_path2 = set(os.listdir(path2))

    return files_in_path1.intersection(files_in_path2)

def helper(path1, path2):
    duplicates = find_duplicate_files(train_left, test_left)
    print("For path1: ", path1, " and path2: ", path2)
    if len(duplicates) == 0:
        print("No duplicate files")
    else:
        print(f"The duplicates file: {', '.join(duplicates)}")


if __name__ == '__main__':
    for path in [train_left, train_right, test_left, test_right]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"path {path} not found")

    helper(train_left, test_left)
    helper(train_left, test_right)

    helper(train_right, test_left)
    helper(train_right, test_right)

    helper(train_left, train_right)
    helper(test_left, test_right)


# We could get all images are unique in the dataset


