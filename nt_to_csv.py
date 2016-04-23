import rdflib
import csv
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid number of parameters!")
        exit(-1)

    g = rdflib.Graph()

    print("-- Loading data")
    g.parse(sys.argv[1], format="nt")

    print("-- Writing data")
    with open(sys.argv[2], mode="w") as out_file:
        writer = csv.writer(out_file)
        for s, p, o in g:
            writer.writerow([s, p, o])

