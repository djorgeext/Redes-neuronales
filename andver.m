% Función para visualizar
function aciertos = andver(s, w)
    [m, ~] = size(s);
    aciertos = 0;
    for i = 1:m
        h = w(1) + s(i, :) * w(2:end)';
        if h >= 0
            y = 1;
        else
            y = -1;
        end
        if all(s(i,:) >= 0)
            yd = 1;
        else
            yd = -1;
        end
        if yd == y
            aciertos = aciertos +1;
        else
            aciertos = aciertos;
        end
        %disp(['entradas ', num2str(s(i, :)), ' salidas ', num2str(y)]);
    end
end

