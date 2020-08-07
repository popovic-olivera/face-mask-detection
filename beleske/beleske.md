# Beleske :thumbsup:

## Dataset

- 2606 slika sa maskama, 1930 slika bez maski
- Na slikama su vecinom Azijati
- Pored pravih slika ljudi sa maskama ima i slika sa generisanim maskama
- Ima u skupu vec transformisanih (rotiranih, sa sumom, flipovanih, skaliranih, mutnih...) slika
- Vecinom su na slikama pojedinacno ljudi, ali ima i mali broj grupnih slika
- Ima i slika koje su skroz naslikane (npr. Iz video igara), ali izgledaju realno
- Maske su razlicitih boja
- Ima i mali broj slika na kojima je pokriveno lice, na primer sakom



## Metrike
 - Confussion matrix 
 - Accuracy (balansirane su klase pa ne bi trebalo da bude problema)
 $$Accuracy = (TP + TN)/(TP + TN + FP + FN)$$
 $$Precision = TP/(TP+FP)$$
 $$Recall = TP/(TP+TN)$$
 $$F-measure = (2* Precision * Recall) / (Precision + Recall)$$
 - ROC Curve 
 $$AUC = (Sp - Np * (Nn +1)/2)/(Np*Nn)$$
 - Sp je suma svih pozitivnih primera