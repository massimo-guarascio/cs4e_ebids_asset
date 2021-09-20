
from ebids_application import *
from misp_connector import *
from mysql_connector import *
from pcap_filter import *
from ebids_application_preprocesser import *
import sklearn as sl
import numpy as np

import os
import time
import json

def extract_tuples_from_dataset(n_tuples: int, dataset : pd.DataFrame, ensemble_pred_y):
    index = 0
    count_inserted = 0
    res = pd.DataFrame()
    while (count_inserted < n_tuples) and (index < dataset.shape[0]):
        if (ensemble_pred_y[index][0] < 0.01) or (str(ensemble_pred_y[index][0]) == 'nan'):
            res = res.append(dataset.loc[index])
            count_inserted = count_inserted+1
        index = index+1
    return res

def send_threat_to_misp(misp_url, misp_key, params):
    misp_connection_params = {
        "misp_url": misp_url,
        "misp_key": misp_key
    }

    send_misp_event(params, misp_connection_params)

def evaluate_accuracy(dataset_path, prediction, delimiter, decimal):
    dataset = readData(dataset_path,delimiter,decimal)
    attack_number = dataset["class"].value_counts()
    data_true = dataset["class"]
    count = 0
    prediction_int = np.around(prediction,0)
    f1 = sl.metrics.f1_score(data_true, prediction_int, average='macro')
    auc_score = sklearn.metrics.roc_auc_score(data_true, prediction)
    pr1, rec1, thr1 = precision_recall_curve(data_true, prediction)
    auc_score_pr = sklearn.metrics.auc(rec1, pr1)




if __name__ == "__main__":

    #Read Parameters from ebids_config.json
    try:
        print("Read Config Parameter")
        fileParams = open("config/ebids_config.json", "r")
        jsonContent = fileParams.read()
        aList = json.loads(jsonContent)
        application_params = aList['ebids_application']
        application_params_base = application_params['base']
    except Exception as e:
        print(e)
        exit(-1)

    normal_tuples_percentage_extraction = application_params_base['normal_tuples_percentage_extraction']
    anomaly_threshold = application_params_base['anomaly_threshold']
    misp_url = application_params_base['misp_url']
    misp_key = application_params_base['misp_key']
    testset_pcap_folder_path = application_params_base['testset_pcap_folder_path']
    testset_csv_folder_path = application_params_base['testset_csv_folder_path']
    data_path = application_params_base['data_path']
    model_folder_path = application_params_base['model_folder_path']
    maximum_threat = application_params_base['maximum_threat']  # If -1 all threats
    delim = application_params_base['delim']
    decimal = application_params_base['decimal']



    application_params_mysql = application_params['mysql']
    mysql_connection_params = application_params_mysql

    last_file_id = 0
    #Read Application Parameters from ebids_ids_store.json
    try:
        print("Read Application Parameter")
        fileParams = open("ebids_ids_store.json", "r")
        jsonContent = fileParams.read()
        application_store = json.loads(jsonContent)
        last_file_id = application_store['last_file_id']
    except Exception as e:
        print(e)
        exit(-1)

    print(" ------ EBIDS IDS Service STARTED --------")

    while True:

        # Automatically select unclassified TEST Dataset for APPLICATION Phase

        testset_csv_path_list = []
        testset_pcap_path_list = []
        directories = os.listdir(testset_csv_folder_path)
        for file in directories:
            complete_csv_file_path = testset_csv_folder_path + os.path.sep + file
            # Find pcap file for CSV
            pcap_name = file[0:-14] + ".pcap"
            complete_pcap_file_path = testset_pcap_folder_path + os.path.sep + pcap_name

            try:
                f = open(complete_pcap_file_path)
                f.close()

                file_from_path = complete_csv_file_path.replace(testset_csv_folder_path, '')
                file_from_path = file_from_path.replace(os.path.sep, '')
                id_file = file_from_path[7:19]
                num = int(id_file)

                if num > last_file_id:
                    testset_csv_path_list.append(complete_csv_file_path)
                    testset_pcap_path_list.append(complete_pcap_file_path)

                    print("Find new CSV to Analyze: " + complete_csv_file_path)
                    print("CSV PCAP FILE: " + complete_pcap_file_path)
                    last_file_id = num

            except IOError as e:
                print("PCAP file not accessible for CSV " + complete_csv_file_path)
                print(e)

        for i in range(len(testset_csv_path_list)):
 
            testset_path = testset_csv_path_list[i]
            pcap_path = testset_pcap_path_list[i]

            preprocess(testset_path, delim, decimal)

            directories = os.listdir(data_path)
            number_of_models = 0
            for file in directories:
                number_of_models = number_of_models + 1

            params = {"file_params": 'ebids_files/default.ini',
                      "model_folder_path": model_folder_path,
                      "preprocesser_folder_path": "ebids_files/out/preprocesser/",
                      "data_path": data_path,
                      "number_of_models": number_of_models,
                      "testset_path": testset_path
                      }

            print("----- Start PREDICTION for " + testset_path + " ----- ")

            ensemble_pred_y = ebids_application(params, False);

            print("----- PREDICTION END ----- ")
            max_value = 0
            for pred in ensemble_pred_y:
                if pred > max_value:
                    max_value = pred[0]


            data = readData(params["testset_path"], delim, decimal)

            if max_value > anomaly_threshold:

                print("----- DETECTED ANOMALY ------");

                # --- LOADING DATASET to Extract JSON With only threat flows---

                count_threat = 0

                index = 0
                for pred in ensemble_pred_y:

                    if count_threat > maximum_threat:
                        if maximum_threat > 0:
                            break

                    if pred[0] > anomaly_threshold:
                        anomalyRows = pd.DataFrame()
                        anomalyRows = anomalyRows.append(data.loc[index])
                        if anomalyRows.shape[0] == 1:
                            anomalyRows.loc[:, 'Anomaly_score'] = pred[0]
                        else:
                            anomalyRows['Anomaly_score'][index] = pred[0]
                        anomaly_details_file_path = 'ebids_files/out/anomalycsv/anomaly.json'
                        anomaly_details_file_name = "anomaly.json"
                        anomalyRows.to_json(anomaly_details_file_path)
                        
                        # Extract IP adress from CSV
                        ipsrc = data.loc[index]['Source_IP']
                        ipdst = data.loc[index]['Destination_IP']
                        timestamp = data.loc[index]['Timestamp']
                        port = data.loc[index]['Destination_Port']
                        source_port = data.loc[index]['Source_Port']

                        # Create PCAP with anomaly flow
                        anomaly_pcap_path = 'ebids_files/out/anomalypcap/anomaly.pcap'
                        print("PCAP Flow Extraction Start")
                        filter_pcap_file(pcap_path, anomaly_pcap_path, ipsrc, ipdst,source_port,port, timestamp)
                        print("PCAP Flow Extraction END")


                        # Send Threat to MISP
                        misp_parameters = {
                            "pcap_file_name": 'anomaly.pcap',
                            "pcap_file_path": anomaly_pcap_path,
                            "ip_dst": [ipdst],
                            "ip_dst_port": [port],
                            "ip_src": [ipsrc],
                            "signature": [],
                            "signature_type": "",
                            "anomaly_score": pred[0],
                            "anomaly_details_file_path": anomaly_details_file_path,
                            "anomaly_details_file_name": anomaly_details_file_name

                        }

                        send_threat_to_misp(misp_url, misp_key, misp_parameters)
                        count_threat = count_threat + 1

                    index = index + 1
            else:
                print("----- NO ANOMALY DETECTED ------");

            print("----- Send Normal Traffic to PCAP DB ------");

            testset_length = len(data)
            testset_length_percentage = (testset_length * normal_tuples_percentage_extraction) / 100

            normal_traffic_dataset = extract_tuples_from_dataset(int(testset_length_percentage), data, ensemble_pred_y)
            normal_traffic_path = 'ebids_files/out/normaltraffic/normal_traffic_dataset.csv'
            normal_traffic_dataset.to_csv(normal_traffic_path)

            print("EXTRACTED NORMAL TRAFFIC LENGTH " + str(normal_traffic_dataset.shape[0]))

            db_params = {
                "pcap_file_path": "",
                "pcap_name": "",
                "csv_file_path": normal_traffic_path,
                "csv_name": "normal_traffic_dataset.csv",
                "anomaly_score": 0
            }

            insert_pcap_on_db(db_params, mysql_connection_params)

            print("----- Normal Traffic sent to DB ------");

        time.sleep(10)

