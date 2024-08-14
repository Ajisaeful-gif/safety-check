// Fungsi untuk membuat video player
function createVideoPlayer(videoId) {
    var container = document.getElementById("video-container");
    container.innerHTML = '<iframe width="100%" height="500px" src="https://www.youtube.com/embed/' + videoId + '?autoplay=1&mute=1" frameborder="0" allowfullscreen></iframe>';
}

// Panggil fungsi createVideoPlayer dengan ID video YouTube yang diinginkan saat halaman terbuka
window.onload = function() {
    createVideoPlayer("aQYoSAzanyA");
};