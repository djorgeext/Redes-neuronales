function [media_ap,p] = grafico(m)

eta = 0.2;
w = unifrnd(-1, 1, 1, m+1);

[s, yd] = entrada_salida(4*m,m);
for i = 1:10
    for j = 1:m
        w = perceptron(s(j, :), eta, yd(j), w);
    end
end
p = zeros(10,1);
media_ap = zeros(100,1);
for g=1:100
    
    for n=1:100
        s_prueba = unifrnd(-1, 1, g, m);
        aciertos = andver(s_prueba, w);
        media_ap(g,1) = media_ap(g,1) + aciertos ;
    end
    media_ap(g,1) = (media_ap(g,1)/100)/(2^(g-1));
    p(g) = g/m;
end

end
