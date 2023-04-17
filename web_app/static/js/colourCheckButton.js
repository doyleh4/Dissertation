function colourButtons(items){
    items.forEach(function(item){
        var btns = document.getElementsByTagName("button");
        btns = Array.prototype.slice.call(btns); // Convert NodeList to array
        btns.forEach(function(btn){
            // Change button background colour
            if (btn.innerHTML == item.Check && item.isMistake){
                btn.style.backgroundColor = "red";
            }
        });
    });
};