

"""
problem in der features extraction - thinking.npy immer nutr leeres array



"""



import os

import mne
import numpy as np

file_name = r"C:\Users\wired\OneDrive\Desktop\Projects\ttt\ttt\workflows\files\Data\KaraOne\filtered_data\MM05"
raw = mne.io.read_raw_fif(fr"{file_name}\raw.fif", preload=True)
data = raw.get_data()


def main():
    output_filename = os.path.join(file_name, "control.txt")
    with open(output_filename, "w") as file:
        file.write(str(data))


def np_main():
    np.set_printoptions(threshold=np.inf)
    filename = r"C:\Users\wired\OneDrive\Desktop\Projects\ttt\ttt\workflows\files\Features\KaraOne\features\P02"
    subject_features = np.load(fr"{filename}\thinking.npy", allow_pickle=True)
    print("subject_features:", subject_features)
    output_filename = os.path.join(filename, "control.txt")
    with open(output_filename, "w") as file:
        file.write(str(subject_features))


if __name__ == "__main__":

    np_main()
