import pandas as pd 
import os
from tqdm import tqdm
import torchaudio

def main():
    
    path = "/work3/s183954/NST_dk/"

    # clean train dataset
    df = pd.read_csv(f"{path}NST_dk.csv")
    df.drop(df[df['text'] == "( ... tavshed under denne indspilning ...)"].index, inplace=True)
    df = df.reset_index()

    counter = 0
    index_list = []
    # iterate over the dataframe
    for index, row in tqdm(df.iterrows()):
        # check if file exists otherwise delete row
        file_name = row['filename_both_channels'] #file_names[item]
        try:
            folder = file_name.split("_")[0] + "/"
        except:
            print(f"File {file_name} does not exist")
            index_list.append(index)
            counter += 1
            continue
        file_path = f"{path}dk/{folder}{file_name.lower()}"
        try:
            torchaudio.load(file_path)
        except:
            index_list.append(index)
            # print(f"File {file_name} does not exist")
            counter += 1
            continue

    df.drop(index_list, inplace=True)
    print(f"Deleted {counter} rows")
    df = df.reset_index()
    df.to_csv(f"/work3/s183954/NST_dk/NST_dk_clean.csv", index=False)


    # clean test dataset
    df = pd.read_csv(f"{path}supplement_dk.csv")
    df.drop(df[df['text'] == "( ... tavshed under denne indspilning ...)"].index, inplace=True)
    df = df.reset_index()
    path += "supplement_dk/testdata/audio/"
    counter = 0
    index_list = []
    # iterate over the dataframe
    for index, row in tqdm(df.iterrows()):
        # check if file exists otherwise delete row
        file_name = row['filename_channel_1']
        try:
            folder = file_name.split("_")[0] + "/"
        except:
            print(f"File {file_name} does not exist")
            index_list.append(index)
            counter += 1
            continue
        file_name = file_name.split('_')[1]
        file_path = f"{path}{folder}{file_name.lower()}"
        try:
            torchaudio.load(file_path)
        except:
            index_list.append(index)
            # print(f"File {file_name} does not exist")
            counter += 1
            continue

    df.drop(index_list, inplace=True)
    print(f"Deleted {counter} rows")
    df = df.reset_index()
    df.to_csv(f"/work3/s183954/NST_dk/supplement_dk_clean.csv", index=False)



if __name__ == "__main__":
    main() 
