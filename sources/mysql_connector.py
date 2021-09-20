import mysql.connector
from datetime import datetime

def insert_pcap_on_db(params, mysql_connection_params):

    mydb = mysql.connector.connect(
      host=mysql_connection_params["host"],
      user=mysql_connection_params["user"],
      password=mysql_connection_params["password"],
      database=mysql_connection_params["database"]
    )

    if params["pcap_file_path"]:
        pcap_file_path = params["pcap_file_path"]
        file_pcap = convertToBinaryData(pcap_file_path)
     else:
        file_pcap = None

    if params["pcap_name"]:
        pcap_name = params["pcap_name"]
    else:
        pcap_name = None

    anomaly_score = params["anomaly_score"]
    csv_file_path = params["csv_file_path"]
    csv_name = params["csv_name"]

    file_csv = convertToBinaryData(csv_file_path)

    mycursor = mydb.cursor()

    data = datetime.now()

    sql = "INSERT INTO pcap(date, pcap_file, pcap_file_name, csv_file, csv_file_name, anomaly_score)  VALUES (%s, %s, %s, %s, %s, %s)"
    val = (data, file_pcap, pcap_name, file_csv, csv_name, anomaly_score)
    mycursor.execute(sql, val)

    mydb.commit()


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData
