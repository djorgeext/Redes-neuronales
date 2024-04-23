% Función de perceptrón
function w = perceptron(inp, eta, yd, w)
    y = 0;
    while yd ~= y
        h = w(1) + inp * w(2:end)';
        if h >= 0
            y = 1;
        else
            y = -1;
        end
        w(1) = w(1) + eta * (yd - y);
        w(2:end) = w(2:end) + eta * (yd - y) * inp;
    end
end