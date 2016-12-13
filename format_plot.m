function format_plot_for_publication (hGraphic)
    hAxis = get (hGraphic, 'CurrentAxes');
    set (hAxis,'FontSize',20); 
    for hLine = get(hAxis, 'Children')
        set (hLine, 'LineWidth', 1.5);
        set (hLine, 'MarkerSize',10);
    end
    set (hGraphic, 'Color','White');
end