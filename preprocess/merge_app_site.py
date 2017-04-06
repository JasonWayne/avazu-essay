import sys

path = sys.argv[1]

with open(path) as f:
    for line in f:
        line.split(" ")
