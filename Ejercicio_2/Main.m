
[media_ap1,p1] = grafico(2);
[media_ap2,p2] = grafico(4);
[media_ap3,p3] = grafico(6);
[media_ap4,p4] = grafico(8);
figure(1), plot(p1,media_ap1,LineWidth=3)
title('Capacidad de aprendizaje del perceptron con 2 entradas')
xlim([0.5 6]);
set(gca, 'FontSize', 16); % Establece el tamaño de fuente en 14 puntos
set(gca, 'LineWidth', 2); % Establece el grosor de los ejes en 2 puntos
xlabel('p/N');
ylabel('C(p,N)/2\^(p-1)');

figure(2), plot(p2,media_ap2,'LineWidth',2)
title('Capacidad de aprendizaje del perceptron con 4 entradas')
xlim([0.5 6]);
set(gca, 'FontSize', 16); % Establece el tamaño de fuente en 14 puntos
set(gca, 'LineWidth', 2); % Establece el grosor de los ejes en 2 puntos
xlabel('p/N');
ylabel('C(p,N)/2\^(p-1)');

figure(3), plot(p3,media_ap3,'LineWidth',3)
title('Capacidad de aprendizaje del perceptron con 6 entradas')
xlim([0.5 6]);
set(gca, 'FontSize', 16); % Establece el tamaño de fuente en 14 puntos
set(gca, 'LineWidth', 2); % Establece el grosor de los ejes en 2 puntos
xlabel('p/N');
ylabel('C(p,N)/2\^(p-1)');

figure(4), plot(p4,media_ap4,'LineWidth',3)
xlim([0.5 6]);
title('Capacidad de aprendizaje del perceptron con 8 entradas')
set(gca, 'FontSize', 16); % Establece el tamaño de fuente en 14 puntos
set(gca, 'LineWidth', 2); % Establece el grosor de los ejes en 2 puntos
xlabel('p/N');
ylabel('C(p,N)/2\^(p-1)', 'Interpreter','latex');
