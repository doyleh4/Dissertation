function uploadFiles() {
    console.log("Upload files to server")
    var foFile = document.getElementById('fo-upload').files[0];
    var dtlFile = document.getElementById('dtl-upload').files[0];

    /* TODO: Update his to have a check to see if both files are filled */
    handleDownload(foFile, dtlFile)


    console.log("Files saves to server")
}

function handleDownload(file, file2){
    let form = new FormData()

    form.append("FO-video", file)
    form.append("DTL-video", file2)

    fetch("/upload_video", {
        method : "POST",
        body : form
    })
    .then(response => response.json())
    .then(data => { console.log("Video download successful")})
    .catch(error => { console.log("Video failed to download")})
}