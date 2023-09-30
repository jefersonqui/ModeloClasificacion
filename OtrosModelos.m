clear;
clc;
%% Entrenamiento clasificador
caracSelec = readtable('caracSelec.txt');
caracSelec_p = readtable('caracSelec_p.txt');
clasificador = fitcauto(caracSelec,'Clase');

%El clasificador evealuó como el modelos svm como el mejor.
predTrain_auto = string(predict(clasificador,caracSelec{:,1:end-1}));
Ttrain_auto = string(caracSelec.Clase);

C_auto = confusionmat(Ttrain_auto,predTrain_auto);
cc_auto = confusionchart(Ttrain_auto,predTrain_auto);
cc_auto.ColumnSummary = 'column-normalized';
cc_auto.RowSummary = 'row-normalized';
cc_auto.Title = 'Matriz de confusión del clasificador(svm auto), conjunto de entrenamiento';

cp = sum(diag(C_auto));%Clases correctamente predichas
t = sum(sum(C_auto));%Número total de muestras
accur_e = (cp/t)*100;


predTest_auto = string(predict(clasificador,caracSelec_p{:,1:end-1}));
Ttest_auto = string(caracSelec_p.Clase);

Cv_auto = confusionmat(Ttest_auto,predTest_auto);
ccv_auto = confusionchart(Ttest_auto,predTest_auto);
ccv_auto.ColumnSummary = 'column-normalized';
ccv_auto.RowSummary = 'row-normalized';
ccv_auto.Title = 'Matriz de confusión del clasificador(svm auto), conjunto de prueba';

cpv = sum(diag(Cv_auto));%Clases correctamente predichas
tv = sum(sum(Cv_auto));%Número total de muestras
accur_v = (cpv/tv)*100;


%Otro modelo
svm_model = fitcecoc(caracSelec,'Clase');

predTrain_svm = string(predict(svm_model,caracSelec{:,1:end-1}));

Cv_auto = confusionmat(Ttrain_auto,predTrain_svm);
cc_svm = confusionchart(Ttrain_auto,predTrain_svm);
cc_svm.ColumnSummary = 'column-normalized';
cc_svm.RowSummary = 'row-normalized';
cc_svm.Title = 'Matriz de confusión del clasificador(svm), conjunto de entrenamiento';

cpv = sum(diag(Cv_auto));%Clases correctamente predichas
tv = sum(sum(Cv_auto));%Número total de muestras
accur_tsvm = (cpv/tv)*100;


predTest_svm = string(predict(svm_model,caracSelec_p{:,1:end-1}));

Cv_auto = confusionmat(Ttest_auto,predTest_svm);
cc_svm = confusionchart(Ttest_auto,predTest_svm);
cc_svm.ColumnSummary = 'column-normalized';
cc_svm.RowSummary = 'row-normalized';
cc_svm.Title = 'Matriz de confusión del clasificador(svm), conjunto de prueba';

cpv = sum(diag(Cv_auto));%Clases correctamente predichas
tv = sum(sum(Cv_auto));%Número total de muestras
accur_testsvm = (cpv/tv)*100;