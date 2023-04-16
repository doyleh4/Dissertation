function colourButtons(items){
    items.forEach(function(item){
        var btns = document.getElementsByTagName("button");
        btns = Array.prototype.slice.call(btns); // Convert NodeList to array
        btns.forEach(function(btn){
            if (btn.innerHTML == item.Check && item.isMistake){
                btn.style.backgroundColor = "red";

                // Add the text for the fix - Doing this here instead of calling a new script in HTML
                /* Add the text to the button - only adds to this instance of the modal */
                // const newElement = document.createElement('div')
                // newElement.innerHTML = "Hello world lol"

                // btn.appendChild(newElement)
            }
        });
    });
};