import subprocess

for i in range(3, 23):
    print "\n\n\nrun field " + str(i) + "\n"
    subprocess.call("python ftrl/ftrl.py train.raw.csv test.raw.csv submission.csv {0}".format(i).split(" "), shell=True)