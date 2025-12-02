import os
import os.path as osp
import tqdm


def generate_index_file(root, filename):
    """Generate index file from the folders."""
    with open(filename, "r") as fp:
        files = [line.strip() for line in fp.readlines() if ".txt" in line]

    for file in tqdm.tqdm(files):
        prefix = osp.splitext(file)[0]
        index_file = osp.join(root, prefix + ".lineidx")
        filein = osp.join(root, file)
        if not osp.exists(filein):
            print(f"{filein} does not exist")
            continue
        if osp.exists(index_file):
            with open(index_file, "r") as fout:
                with open(filein, "r") as fin:
                    if len(fin.readlines()) == len(fout.readlines()):
                        continue
        with open(index_file, "w") as fout:
            with open(filein, "r") as fin:
                fsize = os.fstat(fin.fileno()).st_size
                fpos = 0
                while fpos != fsize:
                    line = fin.readline()
                    fout.write(str(fpos) + "\n")
                    fpos = fin.tell()

    print("done with index generation")


if __name__ == "__main__":
    root = "/data/dataset/all-data-121922"
    filename = "all_data_0802.txt"
    generate_index_file(root, filename)
