CREATE TABLE `differencesmeasuretg` (
  `iddifferencesmeasure` int(11) NOT NULL AUTO_INCREMENT,
  `idSensor` int(11) NOT NULL,
  `difference` float DEFAULT NULL,
  `instant` datetime NOT NULL,
  `diffWith` int(11) NOT NULL,
  PRIMARY KEY (`iddifferencesmeasure`),
  KEY `idx_instant` (`instant`)
) ENGINE=InnoDB AUTO_INCREMENT=6214810 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

CREATE TABLE `anomaliestg` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `idSensor` int(11) NOT NULL,
  `date` datetime NOT NULL,
  `anomaly` tinyint(4) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_date` (`date`)
) ENGINE=InnoDB AUTO_INCREMENT=6029569 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
