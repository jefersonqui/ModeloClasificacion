clear
%% Creacion imageDataStore
path = 'C:\Users\USUARIO\Desktop\IngenieriaFisica\IX Semestre\ProcesamientoOpticoDigitaldeImagenes\PY_FINAL_elementosElectricos\Base de datos Seleccionada';
imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames'); %Creacion ImageDatastore
tbl = countEachLabel(imds); % Numero de imagenes por categoria


%% impresión de las imagenes a trabajar 
figure(1)
montage(imds.Files(1:320:end))

%% preaparación del set de entrenamiento y de validación (t:60%) y
% Aumentar el tamaño del conjunto de entrenamiento (85%)
[trainingSet, validationSet] = splitEachLabel(imds, 0.85, 'randomize');

montage(imds);
montage(trainingSet);
montage(validationSet);

%% Cuenta del numero de etiquetas
T_total = countEachLabel(imds);
T_train = countEachLabel(trainingSet);
T_test = countEachLabel(validationSet);

color = [1 0.8 0; 0 1 0.2; 0 0.5 1; 1 0 1; 0.5 0.3 0.8];

figure('Name','Número de muestras para cada clase');
b1 = bar(T_total{:,1},T_total{:,2},'FaceColor','flat'); title('Numero de muestras totales');
b1.CData = color;
%table2latex(T_total,'NeuralNetwork_WorkSpace\TablaDeDistribucionDatosTotales.tex');

figure('Name','Número de muestras para cada clase, conjunto de entrenamiento');
b2 = bar(T_train{:,1},T_train{:,2},'FaceColor','flat'); title('Numero de muestras para el entrenamiento');
b2.CData = color;
%table2latex(T_train,'NeuralNetwork_WorkSpace\TablaDeDistribucionDatosEntrenamiento.tex');

figure('Name','Número de muestras para cada clase');
b3 = bar(T_test{:,1},T_test{:,2},'FaceColor','flat'); title('Numero de muestras para la prueba');
b3.CData = color;
%table2latex(T_test,'NeuralNetwork_WorkSpace\TablaDeDistribucionDatosPrueba.tex');

%% creación de un modelo de bolsa de características (BOF)
% Ajustar los parámetros del modelo de bolsa de características
% vs(1000),sf(0.8) --> p=85%
bag = bagOfFeatures(trainingSet, 'VocabularySize', 10000, 'StrongestFeatures', 0.4);

% Cálculo de vector de características de img_1 
img = readimage(imds, 1);
featureVector = encode(bag, img);

%% Iimpresión del histograma 'visual word ocurrences'
figure(2)
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

%% clasificador de categorias con BOF
% Experimentar con diferentes algoritmos de clasificación
SVMTemplate = templateSVM('KernelFunction', 'linear');
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag, 'LearnerOptions', SVMTemplate);
%% evaluar el rendimiento del clasificador de categorías con el set de entrenamiento
confMatrixTrain = evaluate(categoryClassifier, trainingSet);

%% evaluar el rendimiento del clasificador de categorías con el set de validación
confMatrixVal = evaluate(categoryClassifier, validationSet);

%% precisión promedio de un clasificador a partir de la matriz de confusión
meanTrain = mean(diag(confMatrixTrain));
meanVal = mean(diag(confMatrixVal));
fprintf('Precisión promedio en el conjunto de entrenamiento: %.2f%%\n', meanTrain * 100);
fprintf('Precisión promedio en el conjunto de validación: %.2f%%\n', meanVal * 100);

%% Implementar el clasificador en una imagen
img = imread('C:\Users\USUARIO\Desktop\IngenieriaFisica\IX Semestre\ProcesamientoOpticoDigitaldeImagenes\PY_FINAL_elementosElectricos\Base de datos Seleccionada\Test\Capacitor electrolitico\r_1246_pcb141rec1_jpg.rf.bd40367e744525fd9a5d6463480f8542.jpg');
figure
imshow(img)

[labelIdx, scores] = predict(categoryClassifier, img);
categoryClassifier.Labels(labelIdx)

%% SAVE
save('BoF_SVM_modelo.mat', 'categoryClassifier');


%% L
%load clasificador_imagenes_70_SVM.mat

%Prediccion matriz de confusión
pred_v = predict(categoryClassifier,validationSet);
T_v = string(validationSet.Labels');

pred_mv(pred_v == 1) = "Capacitor electrolitico";
pred_mv(pred_v == 2) = "Circuito integrado";
pred_mv(pred_v == 3) = "Led";
pred_mv(pred_v == 4) = "Pulsador";
pred_mv(pred_v == 5) = "Reloj";

C = confusionmat(T_v,pred_mv);
c_v = confusionchart(T_v,pred_mv);
c_v.ColumnSummary = 'column-normalized';
c_v.RowSummary = 'row-normalized';
c_v.Title = 'Matriz de confusión de la red neuronal, conjunto de prueba';

cp_v = sum(diag(C));%Clases correctamente predichas
t_v = sum(sum(C));%Número total de muestras
accur_v = (cp_v/t_v)*100;

%prediccion matriz de confusion conjunto de entrenamiento
pred_e = predict(categoryClassifier,trainingSet);
T_e = string(trainingSet.Labels');

pred_me(pred_e == 1) = "Capacitor electrolitico";
pred_me(pred_e == 2) = "Circuito integrado";
pred_me(pred_e == 3) = "Led";
pred_me(pred_e == 4) = "Pulsador";
pred_me(pred_e == 5) = "Reloj";

C_e = confusionmat(T_e,pred_me);
c_e = confusionchart(T_e,pred_me);
c_e.ColumnSummary = 'column-normalized';
c_e.RowSummary = 'row-normalized';
c_e.Title = 'Matriz de confusión de la red neuronal, conjunto de prueba';

cp_e = sum(diag(C));%Clases correctamente predichas
t_e = sum(sum(C));%Número total de muestras
accur_e = (cp_e/t_e)*100;