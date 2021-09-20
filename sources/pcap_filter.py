import socket
import dpkt
import datetime
import time


def tcp_flags(flags):
    ret = ''
    if flags & dpkt.tcp.TH_FIN:
        ret = ret + 'F'
    if flags & dpkt.tcp.TH_SYN:
        ret = ret + 'S'
    if flags & dpkt.tcp.TH_RST:
        ret = ret + 'R'
    if flags & dpkt.tcp.TH_PUSH:
        ret = ret + 'P'
    if flags & dpkt.tcp.TH_ACK:
        ret = ret + 'A'
    if flags & dpkt.tcp.TH_URG:
        ret = ret + 'U'
    if flags & dpkt.tcp.TH_ECE:
        ret = ret + 'E'
    if flags & dpkt.tcp.TH_CWR:
        ret = ret + 'C'

    return ret

def filter_pcap_file(filename_path, new_filename_path, ipsrc, ipdst,source_port, dest_port, timestamp):


    # Open the pcap file
    f = open(filename_path, 'rb')
    wf = open(new_filename_path, 'wb')
    pcap = dpkt.pcap.Reader(f)
    wpcap = dpkt.pcap.Writer(wf, snaplen=110000)

    # We need to reassemble the TCP flows before decoding the HTTP
    # Connections with current buffer

    startFlow = False
    capture = True
    for num, (ts, buf) in enumerate(pcap):

        eth = dpkt.ethernet.Ethernet(buf)
        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
            continue
        ip = eth.data


        if ip.p == dpkt.ip.IP_PROTO_TCP:

            packet = ip.data

            tupl = (socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst), packet.sport, packet.dport, ts)

            if (tupl[0] == ipsrc and tupl[1] == ipdst and tupl[2] == source_port and tupl[3] == dest_port) or (
                    tupl[0] == ipdst and tupl[1] == ipsrc and tupl[2] == dest_port and tupl[3] == source_port):
                timestamp2 = datetime.datetime.fromtimestamp(ts)
                timestamp2_string = timestamp2.strftime('%d/%m/%Y %I:%M:%S %p')
                if timestamp2_string == timestamp and capture == True:
                    startFlow = True

                if startFlow:
                    try:
                        wpcap.writepkt(buf, ts=ts)
                    except:
                        print("EXCEPT")
                        pass

                if ((packet.flags & dpkt.tcp.TH_FIN)):
                     startFlow = False
                    capture = False

                timestamp_start = time.mktime(datetime.datetime.strptime(timestamp,"%d/%m/%Y %I:%M:%S %p").timetuple())
                timestamp_packet = time.mktime(datetime.datetime.strptime(timestamp2_string,"%d/%m/%Y %I:%M:%S %p").timetuple())
                timestamp_dt = timestamp_packet - timestamp_start
                if timestamp_dt > 600:
                    startFlow = False

    wf.flush()
    wf.close()
    f.close()
