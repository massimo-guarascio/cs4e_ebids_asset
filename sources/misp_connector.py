import io
from pymisp import PyMISP
from pymisp import MISPEvent, MISPObject
from datetime import datetime

def send_misp_event(event_params, misp_connection_params):
    print("Send New Event to MISP Server...")

    #----------------- MISP Connection Parameters ----------------

    misp_url = misp_connection_params["misp_url"]
    misp_key = misp_connection_params["misp_key"]
    misp_verifycert = False


    #----------------- EVENT DATA Parameters -----------------------
    creation_date = datetime.now()

    # New Event details fields:
    event_info = "EBIDS Detected Anomaly"
    event_distribution = 0  # Optional, defaults to MISP.default_event_distribution in MISP config
    event_threat_level_id = 2  # Optional, defaults to MISP.default_event_threat_level in MISP config
    event_analysis = 0  # Optional, defaults to 0 (initial analysis)

    #------------------- Security Event Object Creation -----------------------------
    user_defined_obj = MISPObject(name='security_event_object', strict=True,
                                  misp_objects_path_custom='mispevent_testfiles')

    if event_params["ip_src"]:
        for ip in event_params["ip_src"]:
            user_defined_obj.add_attribute('ip_src', value=ip)

    if event_params["ip_dst"]:
        for ip in event_params["ip_dst"]:
            user_defined_obj.add_attribute('ip_dst', value=ip)

    if event_params["ip_dst_port"]:
        for ip in event_params["ip_dst_port"]:
            user_defined_obj.add_attribute('ip_dst_port', value=str(ip))

    user_defined_obj.add_attribute('creation-date', value=creation_date)

    if event_params["signature"]:
        for signature in event_params["signature"]:
            user_defined_obj.add_attribute('signature', value=signature)

    if event_params["signature_type"]:
        user_defined_obj.add_attribute('signature_type', value=event_params["signature_type"])

    #print(event_params["anomaly_score"])
    if event_params["anomaly_score"]:
        anomaly_score = "{\"EBIDS\": {\"version\": \"0.2\", \"reference\": \"https://github.com/massimo-guarascio/cs4e_ebids_asset\", \"attacks\": [{ \"attack_type\": \"ANOMALY\", \"confidence\": \""+str(event_params["anomaly_score"])+"\"}]}}"
        #print("ADD ANOMALY SCORE ")
        user_defined_obj.add_attribute('attack_type', value=anomaly_score)

    def convertToBinaryData(filename):
        # Convert digital data to binary format
        with open(filename, 'rb') as file:
            binaryData = file.read()
            buffer = io.BytesIO(binaryData)
        return buffer


    if event_params["pcap_file_path"]:
        pcap_file_path = event_params["pcap_file_path"]
        pcap_file_name = event_params["pcap_file_name"]
        #print("PCAP FILE PATH " + pcap_file_path)
        file = convertToBinaryData(pcap_file_path)
        user_defined_obj.add_attribute('pcap_file', value=pcap_file_name, data=file, expand='binary')

    if event_params["anomaly_details_file_path"]:
        anomaly_details_file_path = event_params["anomaly_details_file_path"]
        anomaly_details_file_name = event_params["anomaly_details_file_name"]
        #print("Anomaly Detail FILE PATH " + anomaly_details_file_path)
        anomaly_details_file = convertToBinaryData(anomaly_details_file_path)
        user_defined_obj.add_attribute('anomaly_details', value=anomaly_details_file_name, data=anomaly_details_file, expand='binary')


    user_defined_obj.add_attribute('verified', value=False)

    # ------ Connection TO MISP Server --------------------

    misp = PyMISP(misp_url, misp_key, misp_verifycert)

    # ------ EVENT Creation --------------------

    event = MISPEvent()
    event.info = event_info  # Required
    event.distribution = event_distribution
    event.threat_level_id = event_threat_level_id
    event.analysis = event_analysis
    event.add_object(user_defined_obj);

    # ------ SEND EVENT --------------------
    misp.add_event(event, pythonify=True)
