import os


def split_file(filepath, chunk_size=50*1024*1024):
    with open(filepath, 'rb') as f:
        i = 0
        while chunk := f.read(chunk_size):
            with open(f"{filepath}.part_{i:03d}", 'wb') as part:
                part.write(chunk)
            i += 1


def merge_file(filepath):
    with open(filepath, 'wb') as out:
        i = 0
        while os.path.exists(f"{filepath}.part_{i:03d}"):
            with open(f"{filepath}.part_{i:03d}", 'rb') as part:
                out.write(part.read())
            i += 1


if __name__ == '__main__':
    split_file("data/LLCP2024.XPT")
