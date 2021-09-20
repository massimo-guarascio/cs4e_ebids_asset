create schema pcap_db;

CREATE TABLE `pcap` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `uuid` varchar(50) DEFAULT NULL,
  `date` datetime DEFAULT NULL,
  `pcap_file` longblob DEFAULT NULL,
  `pcap_file_name` varchar(45) DEFAULT NULL,
  `csv_file` longblob DEFAULT NULL,
  `csv_file_name` varchar(45) DEFAULT NULL,
  `anomaly_score` decimal(5,4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=289 DEFAULT CHARSET=utf8mb4;