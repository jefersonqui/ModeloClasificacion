clear; 
clc;
%% Carga del modelo de clasificación

path = 'D:\UNICAUCA_2023\PODI\ProyectoFinal\Clasificador_componentes\ClasificadorCompElect_OrtegaAndres_QuiguantarJeferson\Prog\imagenesPrueba';
modeloBOF = load('BoF_SVM_modelo.mat');
imds = imageDatastore(path); %Para seleccionar todas las imágenes
files = imds.Files; %Guardando las rutas a cada imagen de prueba

for i = 1:length(files)
    img = imread(files{1});    
    [pred, score] = predict(modeloBOF.categoryClassifier,img);
    imshow(img);
    if pred == 1
        pred = "Capacitor electrolitico";
    elseif pred == 2
        pred = "Circuito integrado";
    elseif pred == 3
        pred = "Led";
    elseif pred == 4
        pred = "Pulsador";
    elseif pred == 5
        pred = "Reloj";
    else
        pred = "Na";
    end
    title([pred,num2str(score)]);
    keyboard
end
