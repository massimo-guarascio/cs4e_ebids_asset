from ebids_application import *

def port_category(value):
    if value < 1024 :
        return "well_known"
    if value < 49152:
        return "registered"
    return "dynamic"

def preprocess(testset_csv_path, delim, decimal):

    # --- LOADING DATASET ---
    data = readData(testset_csv_path, delim, decimal)
    data["port_type"] = data["Destination_Port"].apply(port_category)

    # Write
    data.to_csv(testset_csv_path, sep=delim, decimal=decimal, index=False)
    #print("Write preprocessed CSV: " + testset_csv_path)