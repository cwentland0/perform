clc
clear all
close all 

%%Begin user input
Galerkin = load("Galerkin_Error_.mat");
OSAB = load("OSAB_Error_.mat");
GalerkinCases = size(Galerkin.error_L2, 3);
OSABCases = size(OSAB.error_L2, 3);
plotIdx = [1, 2, 3, 13, 14, 15];
Plots = ["SigmaL2", "RAE", "SigmaRAE"];   %Options: L2, SigmaL2, RAE, SigmaRAE
%%End user input



if not(length(plotIdx)<=GalerkinCases)
    disp("Plot indexes should be less than the cases investigated")
end

if not(max(plotIdx) <= GalerkinCases)
    disp("Plot within the cases indexes considered")
end


Colors = {'r', [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560], ...
    [0.4660 0.6740 0.1880], [0.3010 0.7450 0.9330], [0.6350 0.0780 0.1840]};

%%State error plots (L2)
if sum(strcmp(Plots, 'L2')) ~= 0
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
        subplot(2, 2, 1)
        hold on
        for ii = 1:length(plotIdx)
            plot(Galerkin.time, Galerkin.error_L2(1, :, ii), '--','color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(plotIdx(ii)), ' (Galerkin - No adaptation)'))
            plot(OSAB.time, OSAB.error_L2(1, :, ii), 'color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(ii), ' (Galerkin - OSAB)'))
        end
        legend()
        set(gca, 'yscale', 'log', 'xscale', 'log')
        box on
        xlabel("$time$", "interpreter", "latex", "fontsize", 15)
        ylabel("$e_{L_{2}}~[Pressure]$", "interpreter", "latex", "fontsize", 15)
        hold off
        subplot(2, 2, 2)
                hold on
        for ii = 1:length(plotIdx)
           plot(Galerkin.time, Galerkin.error_L2(2, :, ii), '--','color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(plotIdx(ii)), ' (Galerkin - No adaptation)'))
            plot(OSAB.time, OSAB.error_L2(2, :, ii), 'color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(ii), ' (Galerkin - OSAB)'))
        end
        legend()
        set(gca, 'yscale', 'log', 'xscale', 'log')
        box on
        xlabel("$time$", "interpreter", "latex", "fontsize", 15)
        ylabel("$e_{L_{2}}~[Velocity]$", "interpreter", "latex", "fontsize", 15)
        hold off
        subplot(2, 2, 3)
                hold on
        for ii = 1:length(plotIdx)
           plot(Galerkin.time, Galerkin.error_L2(3, :, ii), '--','color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(plotIdx(ii)), ' (Galerkin - No adaptation)'))
            plot(OSAB.time, OSAB.error_L2(3, :, ii), 'color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(ii), ' (Galerkin - OSAB)'))
        end
        legend()
        set(gca, 'yscale', 'log', 'xscale', 'log')
        box on
        xlabel("$time$", "interpreter", "latex", "fontsize", 15)
        ylabel("$e_{L_{2}}~[Temperature]$", "interpreter", "latex", "fontsize", 15)
        hold off
        subplot(2, 2, 4)
                hold on
        for ii = 1:length(plotIdx)
            plot(Galerkin.time, Galerkin.error_L2(4, :, ii), '--','color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(plotIdx(ii)), ' (Galerkin - No adaptation)'))
            plot(OSAB.time, OSAB.error_L2(4, :, ii), 'color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(ii), ' (Galerkin - OSAB)'))
        end
        legend()
        set(gca, 'yscale', 'log', 'xscale', 'log')
        box on
        xlabel("$time$", "interpreter", "latex", "fontsize", 15)
        ylabel("$e_{L_{2}}~[Mass~fration]$", "interpreter", "latex", "fontsize", 15)
        hold off
        sgtitle('$L_{2}~error~v.s.~time$', 'interpreter', 'latex', 'fontsize', 15)
        saveas(gcf,'Error_L2.png')
end

%%State Sigma error plots (L2)

if sum(strcmp(Plots, 'SigmaL2')) ~= 0
    figure()
    hold on
    for ii = 1:length(plotIdx)
         plot(Galerkin.time, Galerkin.error_sigma_L2(:, ii), '--','color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(plotIdx(ii)), ' (Galerkin - No adaptation)'))
         plot(OSAB.time, OSAB.error_sigma_L2(:, ii), 'color', Colors{ii}, 'linewidth', 2, 'DisplayName', strcat('Case ', num2str(ii), ' (Galerkin - OSAB)'))
    end
    hold off
    legend('location', 'northwest')
    set(gca, 'yscale', 'log', 'xscale', 'log')
    box on
    xlabel("$time$", "interpreter", "latex", "fontsize", 15)
    ylabel("$\Sigma e_{L_{2}}$", "interpreter", "latex", "fontsize", 15)
    title("$\Sigma e_{L_{2}}~v.s.~time$", "interpreter", "latex", "fontsize", 15)
    set(gcf,'Position',[100 100 800 700])
    saveas(gcf,'Error_Sigma_L2.png')
end





