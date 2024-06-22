function [s, yd] = entrada_salida(x,m)
    s = 2*unifrnd(-1, 1, x, m)-1;
    yd = 2*unifrnd(-1, 1, x, 1)-1;
    % for i = 1:x
    %     if all(s(i, :) >= 0)
    %         yd(i) = 1;
    %     else
    %         yd(i) = -1;
    %     end
    % end
end