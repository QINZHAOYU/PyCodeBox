import os


def collect_files(file_dir, file_list):
    for root, dirs, fnames in os.walk(file_dir):
        for f in fnames:
            arr = f.split('.')
            if len(arr) == 2 and arr[1] == "h":
                file_list.append(os.path.join(root, f))
        for d in dirs:
            collect_files(os.path.join(root, d), file_list)


def rewrite_annotation(f):
    with open(f, "rb") as raw, open(f + ".bak", "wb+") as rep:
        for line in raw.readlines():
            line = line.decode("UTF-8", "ignore")
            if line.find(r"\brief") != -1:
                new_line = line.replace(r"\brief", "@brief")
                new_line = new_line.encode()
                rep.write(new_line)
            elif line.find(r"\param") != -1:
                new_line = line.replace(r"\param", "@param")
                new_line = new_line.encode()
                rep.write(new_line)
            elif line.find(r"\returns") != -1:
                new_line = line.replace(r"\returns", "@returns")
                new_line = new_line.encode()
                rep.write(new_line)
            else:
                line = line.encode()
                rep.write(line)
    os.remove(f)
    os.rename(f+".bak", f)


if __name__ == "__main__":
    file_list = []
    collect_files("./", file_list)

    for f in file_list:
        rewrite_annotation(f)
