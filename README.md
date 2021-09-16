# CS4E - EBIDS asset
EBIDS (Ensemble Based Intrusion Detection System) is a machine learning based classification model for intrusion detection exploiting ensembles of classifiers. Multiple base models are trained on data gathered in different time windows where different types of attacks can occur (Data Chunks). These base classifiers take the form of Deep Neural Networks (DNNs) sharing all the same architecture, but trained against different samples of the given training data. Finally, an incremental learning scheme is adopted to cope with different problems such as Large high-speed datastream and rare attacks.

This implementation is a refactoring and improvement of a preliminary version available at address https://github.com/massimo-guarascio/dnn_ensemble_ids . It is devoted to demonstrating the capabilities of the proposed methodology.

## Authors

The code is developed and maintained by Massimo Guarascio, Gianlugi Folino and Nunziato Cassavia (massimo.guarascio@icar.cnr.it , gianluigi.folino@icar.cnr.it , nunziato.cassavia@icar.cnr.it)

## Video demonstration
A video demonstration of the capabilities of EBIDS, its integration with the TIP (a MISP Network) and cooperation with other security tools (E.g., [NetGen](https://github.com/daniele-canavese/netgen)) is provided in the sub-folder "Video_Demonstration".
Direct link [here](https://github.com/massimo-guarascio/cs4e_ebids_asset/tree/main/Video_Demonstration).
<!---
old video link
https://drive.google.com/file/d/197Vjl2LPmhMkv9oB7OlKwTZ0ayExLaQ7/view
-->

# References
[1] F. Folino, G. Folino, M. Guarascio, F.S. Pisani, L. Pontieri. On learning effective ensembles of deep neural networks for intrusion detection. Information Fusion, 2021. doi: https://doi.org/10.1016/j.inffus.2021.02.007
