function lbp = t_lbp(imds)
    lbp_gray = transform(imds,@rgb2gray);
    ds_lbp = transform(lbp_gray,@extractLBPFeatures);
    data_lbp = readall(ds_lbp);
    [~,c] = size(data_lbp);
    nombres = cell(1,c);
    for i = 1:c
        nombres{i} = ['lbp',num2str(i)];
    end
    Tabla = array2table(data_lbp,'VariableNames',nombres);
    Tabla.Clase = imds.Labels;
    lbp = Tabla;
end

