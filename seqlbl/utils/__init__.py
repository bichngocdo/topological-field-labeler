import subprocess


def file_len(fp):
    p = subprocess.Popen(['wc', '-l', fp],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])
