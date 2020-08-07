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
- U delu sa maskama, od ~2600 slika ~1000 je sa sintetisanim maskama.


## Metrike
 - Confussion matrix 
 - Accuracy (balansirane su klase pa ne bi trebalo da bude problema)
 $$Accuracy = (TP + TN)/(TP + TN + FP + FN)$$
 $$Precision = TP/(TP+FP)$$
 $$Recall = TP/(TP+TN)$$
 $$F-measure = (2* Precision * Recall) / (Precision + Recall)$$
 
 
## O modelu
 - Gde gresi model na slikama:
    - Najvise gresi na **udaljenim slikama** i slikama iz profila. Slike iz profila su cesce u skupu, tako da je kod njih nesto drugo u pitanju. Udaljenih slika ima malo, te je to verovatno pravilnost.
    - Najveca greska se javlja na **slikama na kojima se ne vidi jasno lice**, oci su pokrivene senkom(sesir, naocare). U sustii ne vidi se celo lice.
    - Sintetisane maske nigde ne prave problem.
    - Model takodje gresi na bezveze slikama (crtani filmovi, igrice)
