# MLDM_COVID-19

This project is focused on finding correlation and creating prediction from the data of COVID-19.<br />

The dataSet Used are:<br />
Dataset lombardia Inqunanti e Meteo :https://www.dati.lombardia.it/stories/s/auv9-c2sj<br />
Posizione Stazioni di rilevamento :https://www.dati.lombardia.it/Ambiente/Stazioni-qualit-dell-aria/ib47-atvt<br />
IMPORTANTE :: UNIRE I DATASET DELL'INQUINAMENTO E QUELLI DELLE STAZIONI PER LOCALIZZARE I VALORI<br />
Protezione Civile:https://github.com/pcm-dpc/COVID-19<br />

## IDEA
Si prendono i dati della protezione civile, degli inquinanti e il meteo.<br />
Feature engineering, dal quale si crea un dataset comune a tutti e due i sottogruppi.<br />
A questo punto si suddividono i due sottogruppi.<br />

Si usano 6 giorni random (senza reinserimento) fra i 14 precedenti per predirre il dopo due giorni e il dopo quattro giorni.
Se non funzionasse si aumentano i giorni presi random. Se non funzionasse si butta l'approccio.

## SEED
1 22 777 6654 432145

## ROADMAP
Si calcolano i nuovi positivi come <b>indice "guida"</b>. <br />
Parallelamente vanno calcolati anche: ricoverati_con_sintomi, terapia_intensiva, nuovi_positivi, deceduti, nuovi ospedalizzati e dimessi.
Poi tramite la correlazione calcoliamo dall'indice guida i valori delle feature.
Si considera se è stato più efficace usare la correlazione o predirre tutti gli indici separatamente.<br />

Inoltre si considera la "feature importance della data". <br />

1) decision tree per fare feature importance.<br />
2) si utilizzano: Random Forest, XGB, Boosting.<br />
3) valutiamo quale modello è più preciso.
