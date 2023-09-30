function hog = s_hog(imds,sz,cellsize)
    imds_resize = transform(imds,@(x) imresize(x,sz));
    imds_hog = transform(imds_resize,@(x) extractHOGFeatures(x,'cellsize',cellsize));
    %transformacion en tabla
    data_hog = readall(imds_hog);    
    [~,c] = size(data_hog);
    nombres = cell(1,c);
    for i = 1:c
        nombres{i} = ['hog',num2str(i)];
    end
    Tabla = array2table(data_hog,'VariableNames',nombres);
    Tabla.Clase = imds.Labels;
    hog = Tabla;
end

