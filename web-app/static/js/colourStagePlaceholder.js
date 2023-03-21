function setColourField(data){
    var stages = ["Setup", "Takeaway", "Backswing", "Downswing", "Followthrough"]
    stages.forEach(function(stage) {
        var relevant = data.filter( item => item["Stage"] == stage);
        var mistakes = relevant.filter( item => item["isMistake"] == true);
        
        if (mistakes.length > 0){
            // If mistake in stage colour background red
            var image = document.getElementById(stage);
            image.style.borderColor = "red";
        }
    });
};