async function askQuestion() {
    const question = document.getElementById("userQuestion").value;

    if (!question) {
        alert("Please enter a question!");
        return;
    }

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
    });

    const data = await response.json();
    document.getElementById("response").innerText = "Answer: " + data.answer;

}
