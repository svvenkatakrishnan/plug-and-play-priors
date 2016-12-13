function format_image_for_publication (hGraphic)
    hAxis = get (hGraphic, 'CurrentAxes');
    set (hAxis,'FontSize',20); 
    set (hGraphic, 'Color','White');
end