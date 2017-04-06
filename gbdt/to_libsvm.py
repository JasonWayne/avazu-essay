import sys

in_path = sys.argv[1]
out_path = sys.argv[2]

with open(in_path) as f_in:
    with open(out_path, 'w') as f_out:
        for line in f_out:
            arr = line.split(" ")
            label = int(arr[1])
            feature_ids = arr[2:]
            f_out.write(str(label) + " " + " ".join(map(lambda s: s + ":1", feature_ids)))
