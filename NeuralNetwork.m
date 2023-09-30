clear;
clc;

%% Lectura de las imágenes
path = 'D:\PRO\Dataset\Imagenes_clasificadas';
imagenes = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
[TrainImagenes, TestImagenes] = splitEachLabel(imagenes,0.85,'randomize'); %Se divide en un conjunto de entrenamiento y prueba

T_total = countEachLabel(imagenes);
T_train = countEachLabel(TrainImagenes);
T_test = countEachLabel(TestImagenes);

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


montage(imagenes);
montage(TrainImagenes);
montage(TestImagenes);

%% Extraccion de carcaterísticas
%De entrenamiento 
lbp = t_lbp(TrainImagenes);
hog = s_hog(TrainImagenes,[100,100],[20,20]);
caracteristicas = [lbp(:,1:end-1), hog]; 
%De prueba
lbp_p = t_lbp(TestImagenes);
hog_p = s_hog(TestImagenes,[100,100],[20,20]);
caracteristicas_p = [lbp_p(:,1:end-1), hog_p]; 

%% Selección de características
%Para el entrenamiento
[idx, scores] = fscchi2(caracteristicas,'Clase');
caracSelec = caracteristicas(:,[idx(1:50) end]);

%Para la prueba
caracSelec_p = caracteristicas_p(:,[idx(1:50) end]);
%writetable(caracSelec);
%writetable(caracSelec_p);

%% Visualizacion de características
clases = string(caracteristicas.Clase);
varSelec = string(caracSelec.Properties.VariableNames);
var = string(caracteristicas.Properties.VariableNames);

clases_p = string(caracteristicas_p.Clase);
varSelec_p = string(caracSelec_p.Properties.VariableNames);
var_p = string(caracteristicas_p.Properties.VariableNames);

figure('Name','Características entrenamiento');
parallelcoords(caracteristicas{:,1:end-1},'group',clases,'quantile',0.25,'labels',var(1,1:end-1)); title('Características originales de entrenamiento');
figure('Name','Características entrenamiento seleccionadas');
parallelcoords(caracSelec{:,1:end-1},'group',clases,'quantile',0.25,'labels',varSelec(1,1:end-1)); title('Características seleccionadas de entrenamiento');

figure('Name','Características prueba');
parallelcoords(caracteristicas_p{:,1:end-1},'group',clases_p,'quantile',0.25,'labels',var_p(1,1:end-1)); title('Características originales de prueba');
figure('Name','Características prueba seleccionadas');
parallelcoords(caracSelec_p{:,1:end-1},'group',clases_p,'quantile',0.25,'labels',varSelec_p(1,1:end-1)); title('Características seleccionadasd de prueba');
%% Creacion y entrenamiento de la red neuronal.
M = caracSelec{:,1:end-1}'; %Matriz de características
T = ((caracSelec.Clase == 'Pulsador') + (caracSelec.Clase == 'Capacitor electrolitico')*2 +...
    (caracSelec.Clase == 'Circuito integrado')*3 + (caracSelec.Clase == 'Led')*4 +...
    (caracSelec.Clase == 'Reloj')*5)'; %Vector de objetivos

net = feedforwardnet([50 50 50 20]); %Crea un objeto de red neuronal;
net = configure(net,M,T); %Configuración de los parametro en base a las entradas y salidas de la red.

net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'logsig';
net.layers{4}.transferFcn = 'radbas';

%  Neural Network Transfer Functions.

%     compet - Competitive transfer function.
%     elliotsig - Elliot sigmoid transfer function.
%     hardlim - Positive hard limit transfer function.
%     hardlims - Symmetric hard limit transfer function.
%     logsig - Logarithmic sigmoid transfer function.
%     netinv - Inverse transfer function.
%     poslin - Positive linear transfer function.
%     purelin - Linear transfer function.
%     radbas - Radial basis transfer function.
%     radbasn - Radial basis normalized transfer function.
%     satlin - Positive saturating linear transfer function.
%     satlins - Symmetric saturating linear transfer function.
%     softmax - Soft max transfer function.
%     tansig - Symmetric sigmoid transfer function.
%     tribas - Triangular basis transfer function.
% 

%La funcion que los divide por defecto es 'dividerand'
net.divideParam.trainRatio = 0.7; 
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

%Metricas de evaluación de desempeño de la NN


%Objetivos 
net.trainParam.epochs = 100;
net.trainParam.goal = 0.01;

%Gráficas
plotFn = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotregression', 'plotconfusion',...
    'ploterrcorr', 'plotsomtop', 'plotwb'};
net.PlotFcns = plotFn;
%  
%     plotconfusion  - Plot classification confusion matrix.
%     ploterrcorr    - Plot autocorrelation of error time series.
%     ploterrhist    - Plot error histogram.
%     plotfit        - Plot function fit.
%     plotinerrcorr  - Plot input to error time series cross-correlation.
%     plotperform    - Plot network performance.
%     plotregression - Plot linear regression.
%     plotresponse   - Plot dynamic network time-series response.
%     plotroc        - Plot receiver operating characteristic.
%     plotsomhits    - Plot self-organizing map sample hits.
%     plotsomnc      - Plot Self-organizing map neighbor connections.
%     plotsomnd      - Plot Self-organizing map neighbor distances.
%     plotsomplanes  - Plot self-organizing map weight planes.
%     plotsompos     - Plot self-organizing map weight positions.
%     plotsomtop     - Plot self-organizing map topology.
%     plottrainstate - Plot training state values.
%     plotwb         - Plot Hinton diagrams of weight and bias values.
%  

%Entrenamiento de la red neuronal
redEnt = train(net,M,T);

view(redEnt);

%% Export
%save('NeuralNetwork_modelo.mat','redEnt');

%% Prueba
M_p = caracSelec_p{:,1:end-1}';
T_p = string(caracSelec_p.Clase');

pred = sim(redEnt,M_p); %Datos predecidos 

pred_m(and(pred >= 0.5, pred <= 1.5)) = "Pulsador";
pred_m(1,and(pred > 1.5, pred <= 2.5)) = "Capacitor electrolitico";
pred_m(1,and(pred > 2.5, pred <= 3.5)) = "Circuito integrado";
pred_m(1,and(pred > 3.5, pred <= 4.5)) = "Led";
pred_m(1,and(pred > 4.5, pred <= 5.5)) = "Reloj";
t_pr = T_p(:,1:265);
C = confusionmat(t_pr,pred_m);
cc = confusionchart(t_pr,pred_m(:,1:end));
cc.ColumnSummary = 'column-normalized';
cc.RowSummary = 'row-normalized';
cc.Title = 'Matriz de confusión de la red neuronal, conjunto de prueba';

cp = sum(diag(C));%Clases correctamente predichas
t = sum(sum(C));%Número total de muestras
accur = (cp/t)*100;

%Conjunto de entrenamiento
M = caracSelec{1:1558,1:end-1}';
T = string(caracSelec.Clase');

pred_e = sim(redEnt,M); %Datos predecidos 

pred_me(and(pred_e >= 0.5, pred_e <= 1.5)) = "Pulsador";
pred_me(1,and(pred_e > 1.5, pred_e <= 2.5)) = "Capacitor electrolitico";
pred_me(1,and(pred_e > 2.5, pred_e <= 3.5)) = "Circuito integrado";
pred_me(1,and(pred_e > 3.5, pred_e <= 4.5)) = "Led";
pred_me(1,and(pred_e > 4.5, pred_e <= 5.5)) = "Reloj";

C = confusionmat(T,pred_me);
cc = confusionchart(T,pred_me);
cc.ColumnSummary = 'column-normalized';
cc.RowSummary = 'row-normalized';
cc.Title = 'Matriz de confusión de la red neuronal, conjunto de entrenamiento';

cp = sum(diag(C));%Clases correctamente predichas
t = sum(sum(C));%Número total de muestras
accur_e = (cp/t)*100;