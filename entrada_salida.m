function [s, yd] = entrada_salida(x,m)
    s = unifrnd(-1, 1, x, m);
    yd = zeros(1, x);
    for i = 1:x
        if all(s(i, :) >= 0)
            yd(i) = 1;
        else
            yd(i) = -1;
        end
    end
end