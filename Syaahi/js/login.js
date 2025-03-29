function toggleForm() {
    var loginSection = document.getElementById('loginSection');
    var signupSection = document.getElementById('signupSection');

    if (loginSection.style.display === "none") {
        loginSection.style.display = "block";
        signupSection.style.display = "none";
    } else {
        loginSection.style.display = "none";
        signupSection.style.display = "block";
    }
}
