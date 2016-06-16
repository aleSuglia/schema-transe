from sklearn.cross_validation import train_test_split
import csv
import codecs
import sys
import os

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]

    with codecs.open(dataset_path, 'r', 'utf-8') as in_file:
         reader = csv.reader(in_file)
         X_train, X_test = train_test_split(list(reader), test_size=0.3, random_state=12345)

    with codecs.open(os.sep.join([output_path, 'test.csv']), 'w', 'utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(X_test)

    #X_train, X_valid = train_test_split(X_train, test_size=0.3, random_state=12345)

    with codecs.open(os.sep.join([output_path, 'train.csv']), 'w', 'utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(X_train)

    #with codecs.open(os.sep.join([output_path, "valid.tsv"]), "w", "utf-8") as out_file:
    #    writer = csv.writer(out_file)
    #    writer.writerows(X_valid)

