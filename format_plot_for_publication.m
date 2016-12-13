function format_plot_for_publication (hGraphic)

    hAxis = get (hGraphic, 'CurrentAxes');
    set (hAxis,'FontSize',20);%20 

    for hLine = get(hAxis, 'Children')
        set (hLine, 'LineWidth', 2);%3
        set (hLine, 'MarkerSize',4);%18
    end
    
    set (hGraphic, 'Color','White');
end
