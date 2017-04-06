import sys

in_path = sys.argv[1]
out_path = sys.argv[2]

filtered_ids = set([999985, 999984, 999990, 999986, 999981, 999980, 999983, 999982, 999976, 999977, 999989, 999987, 999988, 999978, 999979])

with open(in_path) as f_in:
    with open(out_path, 'w') as f_out:
        for line in f_in:
            arr = line.strip().split(" ")
            label = int(arr[1])
            feature_ids = sorted(filter(lambda x: x not in filtered_ids, map(lambda x: int(x), arr[2:])))
            f_out.write(str(label) + " " + " ".join(map(lambda s: str(s) + ":1", feature_ids)) + "\n")
