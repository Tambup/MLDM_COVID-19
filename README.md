# MLDM_COVID-19

This project is focused on finding correlation and creating prediction from the data of COVID-19.<br />

The DataSheet Used are:<br />
Dataset lombardia Inqunanti e Meteo :https://www.dati.lombardia.it/stories/s/auv9-c2sj<br />
Posizione Stazioni di rilevamento :https://www.dati.lombardia.it/Ambiente/Stazioni-qualit-dell-aria/ib47-atvt<br />
IMPORTANTE :: UNIRE I DATASET DELL'INQUINAMENTO E QUELLI DELLE STAZIONI PER LOCALIZZARE I VALORI<br />
Protezione Civile:https://github.com/pcm-dpc/COVID-19<br />

## IDEA
Si prendono i dati della protezione civile, degli inquinanti e il meteo.<br />
Feature engineering, dal quale si crea un dataset comune a tutti e due i sottogruppi.<br />
A questo punto si suddividono i due sottogruppi.<br />

### GRUPPO ROVETTA TAMBURINI


### GRUPPO BERARDI TOMASETTI





Si usano 5 giorni random (senza reinserimento) fra i 14 precedenti per predirre il dopo due giorni e il dopo quattro giorni.
Se non funzionasse si aumentano i giorni presi random. Se non funzionasse si butta l'approccio.
