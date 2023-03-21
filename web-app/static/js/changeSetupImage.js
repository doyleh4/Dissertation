function changeImage(path, id, button, data){
    // This script only works for dtl images atm
    var img = document.getElementById(id);

    img.src = path;

    // Get the "fix" element to change the drill
    var field = document.getElementsByClassName("fix_text") 
    field = Array.prototype.slice.call(field); // Convert NodeList to array

    // button is the button that was clicked, which has innerHTML of the check 
    // Yes this is doing it for all modals. TODO: Need to fix this at some point
    field.forEach(function(f){
        data.forEach(function(item){
            if (button.innerHTML == item.Check){
                f.innerHTML = item.Fix;
                f.style.backgroundCOlor = "white"
            };
        });
    });
    
    var arr = ["Setup img", "Takeaway img", "Backswing img", "Takeaway img"]
    // if button hit was to retrun back to stage image - delete the text in field
    field.forEach(function(f){
        arr.forEach(function(stage){
            if(button.innerHTML == stage){
                f.innerHTML = "";
            }
        });
    });

}