
[media_ap1,p1] = grafico(2);
[media_ap2,p2] = grafico(3);
[media_ap3,p3] = grafico(4);
[media_ap4,p4] = grafico(5);
plot(p1,media_ap1,'r',LineWidth=3)
hold on
plot(p2,media_ap2,'b','LineWidth',3)
plot(p3,media_ap3,'k','LineWidth',3)
plot(p4,media_ap4,'g','LineWidth',3)
set(gca, 'FontSize', 16); % Establece el tamaño de fuente en 14 puntos
set(gca, 'LineWidth', 2); % Establece el grosor de los ejes en 2 puntos
xlim([0 10]);
xlabel('p/N');
ylabel('C(p,N)/2\^(p-1)');